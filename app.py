import random
import math
import copy
import os
import signal
import json
import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# Korean Font Setup (Note: May require a system font like 'Malgun Gothic' to be installed)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
interrupted = False
LAYOUT_DATA_FILE = "optimized_layout_data.json" # File name for data exchange

# --- Existing Utility Functions (Adapted from PPO code) ---
class Machine:
    def __init__(self, id, name, footprint, cycle_time=0, clearance=0, wall_affinity=False):
        self.id = id; self.name = name; self.footprint = footprint; self.cycle_time = cycle_time
        self.clearance = clearance; self.wall_affinity = wall_affinity; self.position = None

# Machine Definitions (Added zone_group for constraint testing)
machines_definitions = [
    {"id": 0, "name": "Raw Material Input", "footprint": (2, 2), "cycle_time": 20, "clearance": 1, "zone_group": None},
    {"id": 1, "name": "1st Cutting", "footprint": (3, 3), "cycle_time": 35, "clearance": 1, "zone_group": 1}, # Zone 1
    {"id": 2, "name": "Milling Process", "footprint": (4, 2), "cycle_time": 45, "clearance": 1, "zone_group": 1}, # Zone 1
    {"id": 3, "name": "Drilling", "footprint": (2, 2), "cycle_time": 25, "clearance": 1, "zone_group": 1}, # Zone 1
    {"id": 4, "name": "Heat Treatment A", "footprint": (3, 4), "cycle_time": 70, "clearance": 2, "zone_group": None},
    {"id": 5, "name": "Precision Machining A", "footprint": (3, 2), "cycle_time": 40, "clearance": 1, "zone_group": 2}, # Zone 2
    {"id": 6, "name": "Assembly A", "footprint": (2, 3), "cycle_time": 55, "clearance": 2, "zone_group": 3}, # Zone 3
    {"id": 7, "name": "Final Inspection A", "footprint": (1, 2), "cycle_time": 15, "clearance": 1, "zone_group": 3}, # Zone 3
    {"id": 8, "name": "2nd Cutting", "footprint": (3, 2), "cycle_time": 30, "clearance": 1, "zone_group": 1}, # Zone 1
    {"id": 9, "name": "Surface Treatment", "footprint": (2, 4), "cycle_time": 50, "clearance": 2, "zone_group": None},
    {"id": 10, "name": "Washing Process 1", "footprint": (2, 2), "cycle_time": 20, "clearance": 1, "zone_group": 2}, # Zone 2
    {"id": 11, "name": "Heat Treatment B", "footprint": (4, 4), "cycle_time": 75, "clearance": 2, "zone_group": None},
    {"id": 12, "name": "Precision Machining B", "footprint": (2, 3), "cycle_time": 42, "clearance": 1, "zone_group": 2}, # Zone 2
    {"id": 13, "name": "Component Assembly", "footprint": (3, 3), "cycle_time": 60, "clearance": 1, "zone_group": 3}, # Zone 3
    {"id": 14, "name": "Quality Inspection B", "footprint": (2, 1), "cycle_time": 18, "clearance": 1, "zone_group": 3}, # Zone 3
    {"id": 15, "name": "Packaging Line A", "footprint": (4, 3), "cycle_time": 30, "clearance": 2, "zone_group": None},
]

# Process Sequence Definition
all_machine_ids_for_sequence = list(range(len(machines_definitions)))
random.shuffle(all_machine_ids_for_sequence)
PROCESS_SEQUENCE = all_machine_ids_for_sequence
# PROCESS_SEQUENCE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 

# Factory Size Definition (Can be changed based on the number of facilities)
FACTORY_WIDTH = 28
FACTORY_HEIGHT = 28

# Target Production and Time Constants
TARGET_PRODUCTION_PER_HOUR = 35
SECONDS_PER_HOUR = 3600
MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND = 0.5


# --- GA Hyperparameters ---
POPULATION_SIZE = 300
NUM_GENERATIONS = 300
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 5
TOURNAMENT_SIZE = 5

# Fitness Function Weights
FITNESS_THROUGHPUT_WEIGHT = 1.0          # Maximize TPH
FITNESS_DISTANCE_WEIGHT = 0.005          # Minimize Euclidean Distance
FITNESS_ZONE_PENALTY_WEIGHT = 0.5        # Zone Constraint Penalty (NEW)
FITNESS_MHS_TURN_PENALTY_WEIGHT = 0.001  # MHS Turn Penalty (NEW)
FITNESS_UTILIZATION_BONUS_WEIGHT = 0.1   # Area Utilization Bonus (NEW)
BONUS_FOR_TARGET_ACHIEVEMENT_FACTOR = 0.2

# New Constraint Parameters
ZONE_1_TARGET_MACHINES = [1, 2, 3, 8] # Cutting, Milling, Drilling
ZONE_1_MAX_SPREAD_DISTANCE = 12.0 # Max Euclidean distance between any two Zone 1 machines
AREA_UTIL_MIN_THRESHOLD = 0.40 # Target 40% utilization (footprint area / total area)


# --- Core GA Utility Functions ---

def initialize_layout_grid(width, height):
    return [[-1 for _ in range(height)] for _ in range(width)]

def can_place_machine(grid, machine_footprint, machine_clearance, x, y, factory_w, factory_h):
    m_width, m_height = machine_footprint
    if not (0 <= x and x + m_width <= factory_w and 0 <= y and y + m_height <= factory_h):
        return False
    check_x_start = x - machine_clearance
    check_x_end = x + m_width + machine_clearance
    check_y_start = y - machine_clearance
    check_y_end = y + m_height + machine_clearance
    for i in range(max(0, check_x_start), min(factory_w, check_x_end)):
        for j in range(max(0, check_y_start), min(factory_h, check_y_end)):
            is_machine_body = (x <= i < x + m_width) and (y <= j < y + m_height)
            if grid[i][j] != -1:
                if is_machine_body:
                    return False
                elif machine_clearance > 0:
                    return False
    return True

def place_machine_on_grid(grid, machine_id, machine_footprint, x, y):
    m_width, m_height = machine_footprint
    for i in range(x, x + m_width):
        for j in range(y, y + m_height):
            grid[i][j] = machine_id

def get_machine_cycle_time(machine_id, all_machines_data):
    for m_def in all_machines_data:
        if m_def["id"] == machine_id:
            return m_def["cycle_time"]
    return float('inf')

def create_individual(machines_in_processing_order_defs, factory_w, factory_h):
    """Generates a valid, initial layout chromosome."""
    chromosome = []
    grid = initialize_layout_grid(factory_w, factory_h)
    for machine_def in machines_in_processing_order_defs:
        valid_placements = []
        m_footprint, m_clearance = machine_def["footprint"], machine_def.get("clearance", 0)
        for x in range(factory_w - m_footprint[0] + 1):
            for y in range(factory_h - m_footprint[1] + 1):
                if can_place_machine(grid, m_footprint, m_clearance, x, y, factory_w, factory_h):
                    valid_placements.append((x, y))
        if not valid_placements: chromosome.append((-1, -1))
        else:
            chosen_x, chosen_y = random.choice(valid_placements)
            chromosome.append((chosen_x, chosen_y))
            place_machine_on_grid(grid, machine_def["id"], m_footprint, chosen_x, chosen_y)
    return chromosome

def calculate_manhattan_distance(pos1, pos2):
    """Calculates Manhattan distance between two machine center points."""
    return abs(pos1['center_x'] - pos2['center_x']) + abs(pos1['center_y'] - pos2['center_y'])

def calculate_total_distance(machine_positions, process_sequence, distance_type='euclidean'):
    """Calculates total travel distance along the process sequence."""
    total_distance = 0
    if not machine_positions or len(process_sequence) < 2: return float('inf')
    
    for i in range(len(process_sequence) - 1):
        m1_id, m2_id = process_sequence[i], process_sequence[i+1]
        pos1 = machine_positions.get(m1_id)
        pos2 = machine_positions.get(m2_id)
        
        if not pos1 or not pos2: return float('inf')
        
        dx = pos1['center_x'] - pos2['center_x']
        dy = pos1['center_y'] - pos2['center_y']
        
        if distance_type == 'manhattan':
            distance = abs(dx) + abs(dy)
        else: # euclidean
            distance = math.sqrt(dx**2 + dy**2)
            
        total_distance += distance
    return total_distance

def calculate_area_metrics(machine_positions, machines_defs_ordered_by_proc_seq, factory_w, factory_h):
    """Calculates total footprint area and the utilization ratio."""
    total_footprint_area = 0
    
    for machine_def in machines_defs_ordered_by_proc_seq:
        if machine_def["id"] in machine_positions:
            w, h = machine_def["footprint"]
            total_footprint_area += w * h
            
    factory_area = factory_w * factory_h
    utilization_ratio = total_footprint_area / factory_area
    
    return total_footprint_area, utilization_ratio

def calculate_zone_penalty(machine_positions, zone_target_ids, max_spread_dist):
    """Calculates penalty for machines violating the zone proximity constraint."""
    zone_machines_pos = [machine_positions[m_id] for m_id in zone_target_ids if m_id in machine_positions]
    
    if len(zone_machines_pos) < 2: return 0.0 # Not enough machines to check
    
    max_dist = 0.0
    for i in range(len(zone_machines_pos)):
        for j in range(i + 1, len(zone_machines_pos)):
            pos1 = zone_machines_pos[i]
            pos2 = zone_machines_pos[j]
            dist = math.sqrt((pos1['center_x'] - pos2['center_x'])**2 + (pos1['center_y'] - pos2['center_y'])**2)
            if dist > max_dist: max_dist = dist
            
    if max_dist > max_spread_dist:
        # Exponential penalty for severe violation
        penalty = (max_dist - max_spread_dist) ** 2
        return penalty
    return 0.0

def calculate_machine_utilization_and_bottleneck(machine_positions, process_sequence, all_machines_data, travel_speed):
    """
    Calculates stage times, finds the bottleneck, and determines utilization for all machines.
    Returns: (max_stage_time, utilization_data)
    """
    if not machine_positions or not process_sequence: return (0.0, {})
    
    stage_times = {}
    
    # 1. Calculate time for each stage (Machine Cycle Time + Travel Time to Next)
    for i in range(len(process_sequence)):
        current_machine_id = process_sequence[i]
        machine_cycle_time = get_machine_cycle_time(current_machine_id, all_machines_data)
        
        travel_time_to_next = 0.0
        if i < len(process_sequence) - 1:
            next_machine_id = process_sequence[i+1]
            pos_curr = machine_positions.get(current_machine_id)
            pos_next = machine_positions.get(next_machine_id)
            
            if pos_curr and pos_next:
                distance = math.sqrt((pos_curr['center_x'] - pos_next['center_x'])**2 + (pos_curr['center_y'] - pos_next['center_y'])**2)
                if travel_speed > 0: 
                    travel_time_to_next = distance / travel_speed
                else: 
                    travel_time_to_next = float('inf')

        # The stage time is the time the *entire line* must wait for this machine + travel to clear.
        current_stage_total_time = machine_cycle_time + travel_time_to_next
        stage_times[current_machine_id] = current_stage_total_time

    # 2. Find the bottleneck (maximum stage time)
    max_stage_time = max(stage_times.values()) if stage_times else 0.0
    if max_stage_time <= 0 or max_stage_time == float('inf'):
        return (0.0, {})

    # 3. Calculate utilization for each machine
    utilization_data = {}
    for i, machine_id in enumerate(process_sequence):
        machine_cycle_time = get_machine_cycle_time(machine_id, all_machines_data)
        
        # Utilization is the ratio of machine's cycle time to the line's cycle time (bottleneck time)
        # We use the machine's cycle time as the resource usage time
        if machine_cycle_time == float('inf') or max_stage_time == 0.0:
            utilization_data[machine_id] = 0.0
        else:
            utilization = machine_cycle_time / max_stage_time
            utilization_data[machine_id] = min(1.0, utilization) # Cap at 100%
            
    return (max_stage_time, utilization_data)


def calculate_fitness(chromosome, machines_defs_ordered_by_proc_seq, process_seq_ids,
                      factory_w, factory_h, target_prod_throughput, material_travel_speed):
    
    grid = initialize_layout_grid(factory_w, factory_h)
    machine_positions = {}
    all_machines_placed_successfully = True
    
    # Ensure machine_utilization is a dictionary, even if empty initially
    result = {"fitness": -float('inf'), "distance": float('inf'), "throughput": 0.0, 
              "is_valid": False, "zone_penalty": 0.0, "mhs_turn_penalty": 0.0, 
              "utilization_ratio": 0.0, "machine_utilization": {}}

    if len(chromosome) != len(machines_defs_ordered_by_proc_seq): return result

    for i, machine_def in enumerate(machines_defs_ordered_by_proc_seq):
        pos_x, pos_y = chromosome[i]
        if pos_x == -1 and pos_y == -1: all_machines_placed_successfully = False; break
        
        m_footprint, m_clearance, m_id = machine_def["footprint"], machine_def.get("clearance", 0), machine_def["id"]
        
        if not can_place_machine(grid, m_footprint, m_clearance, pos_x, pos_y, factory_w, factory_h):
            all_machines_placed_successfully = False; break
            
        place_machine_on_grid(grid, m_id, m_footprint, pos_x, pos_y)
        machine_positions[m_id] = {"x": pos_x, "y": pos_y, "center_x": pos_x + m_footprint[0]/2.0, "center_y": pos_y + m_footprint[1]/2.0}

    if not all_machines_placed_successfully or len(machine_positions) != len(machines_defs_ordered_by_proc_seq):
        return result 

    # 1. Throughput & Machine Utilization 
    max_stage_time, utilization_data = calculate_machine_utilization_and_bottleneck(
        machine_positions, process_seq_ids, machines_definitions, material_travel_speed)
    
    result["machine_utilization"] = utilization_data
    throughput = SECONDS_PER_HOUR / max_stage_time if max_stage_time > 0 and max_stage_time != float('inf') else 0.0
    
    # Check if calculation failed
    if max_stage_time == float('inf') or throughput == 0.0: return result

    # 2. Euclidean Distance
    total_euclidean_dist = calculate_total_distance(machine_positions, process_seq_ids, distance_type='euclidean')

    # 3. Zone Constraint Penalty (NEW)
    zone_penalty = calculate_zone_penalty(machine_positions, ZONE_1_TARGET_MACHINES, ZONE_1_MAX_SPREAD_DISTANCE)
    result["zone_penalty"] = zone_penalty
    
    # 4. Area Utilization Bonus (NEW)
    _, utilization_ratio = calculate_area_metrics(machine_positions, machines_defs_ordered_by_proc_seq, factory_w, factory_h)
    result["utilization_ratio"] = utilization_ratio
    utilization_bonus = 0.0
    if utilization_ratio >= AREA_UTIL_MIN_THRESHOLD:
        utilization_bonus = (utilization_ratio - AREA_UTIL_MIN_THRESHOLD) * FACTORY_WIDTH * FACTORY_HEIGHT * FITNESS_UTILIZATION_BONUS_WEIGHT

    # 5. MHS Turn Penalty (Approximation using Manhattan distance) (NEW)
    total_manhattan_dist = calculate_total_distance(machine_positions, process_seq_ids, distance_type='manhattan')
    mhs_turn_penalty_value = total_manhattan_dist * FITNESS_MHS_TURN_PENALTY_WEIGHT 
    result["mhs_turn_penalty"] = mhs_turn_penalty_value

    # Combine fitness components
    fitness_val = (FITNESS_THROUGHPUT_WEIGHT * throughput) \
                - (FITNESS_DISTANCE_WEIGHT * total_euclidean_dist) \
                - (FITNESS_ZONE_PENALTY_WEIGHT * zone_penalty) \
                - mhs_turn_penalty_value \
                + utilization_bonus
                
    if throughput >= target_prod_throughput: 
        fitness_val += throughput * BONUS_FOR_TARGET_ACHIEVEMENT_FACTOR
    
    result["fitness"] = fitness_val
    result["distance"] = total_euclidean_dist
    result["throughput"] = throughput
    result["is_valid"] = True
    return result


def print_layout(grid, machine_positions):
    print("--- Current Layout ---")
    for row_idx_actual in range(FACTORY_HEIGHT -1, -1, -1):
        row_to_print = [f"{grid[col_idx_actual][row_idx_actual]:2d}" if grid[col_idx_actual][row_idx_actual] != -1 else "__" for col_idx_actual in range(FACTORY_WIDTH)]
        print(f"Y{row_idx_actual:<2}| " + " ".join(row_to_print))
    header = "    " + " ".join(f"X{i:<2}" for i in range(FACTORY_WIDTH))
    print("-" * len(header)); print(header)
    if not machine_positions: print("No facilities placed"); return
    print("\n--- Machine Positions (Top-Left, Center) ---")
    for machine_id in PROCESS_SEQUENCE:
        if machine_id in machine_positions:
            pos_data = machine_positions[machine_id]
            machine_def = next((m for m in machines_definitions if m["id"] == machine_id), None)
            machine_name = machine_def["name"] if machine_def else "Unknown"
            print(f"Machine ID {machine_id} ({machine_name}): Top-Left ({pos_data['x']}, {pos_data['y']}), Center ({pos_data['center_x']:.1f}, {pos_data['center_y']:.1f})")

def visualize_layout_plt(grid_layout_to_show, machine_positions_map, factory_w, factory_h, process_sequence_list, machine_definitions_list, filename="ga_best_layout.png"):
    """Visualizes the final layout using matplotlib and saves it to a file."""
    fig, ax = plt.subplots(1, figsize=(factory_w/2, factory_h/2 + 1)) 
    ax.set_xlim(-0.5, factory_w - 0.5)
    ax.set_ylim(-0.5, factory_h - 0.5)
    ax.set_xticks(range(factory_w))
    ax.set_yticks(range(factory_h))
    ax.set_xticklabels(range(factory_w))
    ax.set_yticklabels(range(factory_h))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() 

    cmap = plt.colormaps.get_cmap('viridis')
    num_machines = len(machine_definitions_list)
    machines_dict_by_id = {m['id']: m for m in machine_definitions_list}

    # --- Draw Zone Constraint Boundary (NEW) ---
    zone_1_centers = [machine_positions_map[m_id] for m_id in ZONE_1_TARGET_MACHINES if m_id in machine_positions_map]
    if len(zone_1_centers) > 0:
        center_x_coords = [c['center_x'] for c in zone_1_centers]
        center_y_coords = [c['center_y'] for c in zone_1_centers]
        
        # Calculate bounding box of centers
        min_cx, max_cx = min(center_x_coords), max(center_x_coords)
        min_cy, max_cy = min(center_y_coords), max(center_y_coords)
        
        # Determine the maximum required spread (x or y) based on the farthest center from the center of the bounding box
        avg_cx, avg_cy = (min_cx + max_cx) / 2, (min_cy + max_cy) / 2
        
        max_dist_from_avg_sq = 0
        for cx, cy in zip(center_x_coords, center_y_coords):
            dist_sq = (cx - avg_cx)**2 + (cy - avg_cy)**2
            if dist_sq > max_dist_from_avg_sq:
                max_dist_from_avg_sq = dist_sq
        
        # The required radius is ZONE_1_MAX_SPREAD_DISTANCE / 2
        required_radius = ZONE_1_MAX_SPREAD_DISTANCE / 2.0
        
        # Draw the required boundary circle (farthest distance between any two centers is 2*radius)
        circle_required = patches.Circle((avg_cx - 0.5, avg_cy - 0.5), 
                                         radius=required_radius, 
                                         linewidth=2, 
                                         edgecolor='red', 
                                         facecolor='none', 
                                         linestyle='--', 
                                         alpha=0.4,
                                         label=f"Zone 1 Max Spread ({ZONE_1_MAX_SPREAD_DISTANCE})")
        ax.add_patch(circle_required)


    # --- Draw Machines ---
    for machine_id_in_seq in process_sequence_list:
        if machine_id_in_seq in machine_positions_map:
            pos_data = machine_positions_map[machine_id_in_seq]
            machine_info = machines_dict_by_id.get(machine_id_in_seq)

            if machine_info:
                x, y = pos_data['x'], pos_data['y']
                width, height = machine_info['footprint']
                clearance = machine_info.get('clearance', 0)
                
                color_value = machine_id_in_seq / max(num_machines - 1, 1) 
                rect_body = patches.Rectangle((x - 0.5, y - 0.5), width, height,
                                              linewidth=1.5, edgecolor='black',
                                              facecolor=cmap(color_value), alpha=0.7)
                ax.add_patch(rect_body)

                text_x = x + width / 2 - 0.5
                text_y = y + height / 2 - 0.5
                ax.text(text_x, text_y, f"M{machine_id_in_seq}\n({machine_info['name'][:5]}..)",
                        ha='center', va='center', fontsize=6, color='white', weight='bold')
                
                if clearance > 0:
                    rect_clearance = patches.Rectangle(
                        (x - clearance - 0.5, y - clearance - 0.5),
                        width + 2 * clearance, height + 2 * clearance,
                        linewidth=1, edgecolor=cmap(color_value),
                        facecolor='none', linestyle=':', alpha=0.5
                    )
                    ax.add_patch(rect_clearance)

    plt.title("Optimized Factory Layout (GA) with Zone Constraint", fontsize=12)
    plt.xlabel("Factory Width (X)")
    plt.ylabel("Factory Height (Y)")
    plt.gca().invert_yaxis() 

    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Final Layout Visualization Image Saved: {filename}")
    except Exception as e:
        print(f"Error occurred while saving layout image: {e}")
    plt.close(fig) 

def selection(population_with_eval_results, tournament_size): 
    if not population_with_eval_results: return None
    actual_tournament_size = min(tournament_size, len(population_with_eval_results))
    if actual_tournament_size == 0: return None
    tournament = random.sample(population_with_eval_results, actual_tournament_size)
    winner = max(tournament, key=lambda item: item[1]['fitness']) 
    return winner[0] 

def crossover(parent1_chromo, parent2_chromo, crossover_rate):
    child1, child2 = list(parent1_chromo), list(parent2_chromo)
    if random.random() < crossover_rate:
        num_genes = len(parent1_chromo)
        if num_genes > 1:
            cut_point = random.randint(1, num_genes - 1)
            child1 = parent1_chromo[:cut_point] + parent2_chromo[cut_point:]
            child2 = parent2_chromo[:cut_point] + parent1_chromo[cut_point:]
    return child1, child2

def mutate(chromosome, machines_in_proc_order_defs, factory_w, factory_h, mutation_rate_per_gene=0.1):
    mutated_chromosome = list(chromosome)
    num_genes = len(chromosome)
    if num_genes == 0: return mutated_chromosome
    for i in range(num_genes):
        if random.random() < mutation_rate_per_gene:
            machine_def_to_mutate = machines_in_proc_order_defs[i]
            m_footprint, m_clearance = machine_def_to_mutate["footprint"], machine_def_to_mutate.get("clearance", 0)
            temp_grid = initialize_layout_grid(factory_w, factory_h)
            for k in range(num_genes):
                if k == i: continue
                prev_pos_x, prev_pos_y = mutated_chromosome[k]
                if prev_pos_x == -1 and prev_pos_y == -1: continue
                other_machine_def = machines_in_proc_order_defs[k]
                place_machine_on_grid(temp_grid, other_machine_def["id"], other_machine_def["footprint"], prev_pos_x, prev_pos_y)
            valid_new_placements = []
            for x_coord in range(factory_w - m_footprint[0] + 1):
                for y_coord in range(factory_h - m_footprint[1] + 1):
                    if can_place_machine(temp_grid, m_footprint, m_clearance, x_coord, y_coord, factory_w, factory_h):
                        valid_new_placements.append((x_coord,y_coord))
            if valid_new_placements: mutated_chromosome[i] = random.choice(valid_new_placements)
    return mutated_chromosome

def signal_handler(signum, frame):
    global interrupted
    print("\n\n⚠️ CTRL+C Detected! Saving graph after current generation completes and exiting...")
    interrupted = True

# [NEW] JSON Saving Function (Updated to include final metrics)
def save_layout_data(machine_positions_map, process_sequence, machines_definitions, factory_w, factory_h, final_metrics, filename):
    """Saves the best layout's data and final metrics to a JSON file for pathfinding analysis."""
    # Convert machine_utilization keys to strings for JSON compatibility
    if 'machine_utilization' in final_metrics:
        final_metrics['machine_utilization'] = {str(k): v for k, v in final_metrics['machine_utilization'].items()}
        
    machine_positions_str_keys = {str(k): v for k, v in machine_positions_map.items()}
    
    data = {
        "factory_width": factory_w,
        "factory_height": factory_h,
        "process_sequence": process_sequence,
        "machines_definitions": machines_definitions,
        "machine_positions_map": machine_positions_str_keys,
        "final_metrics": final_metrics # NEW
    }
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Best layout data saved to {filename} for pathfinding.")
    except Exception as e:
        print(f"Error saving layout data to JSON: {e}")

# ----------------------------------------------------------------------------------------------------
# ----------------------------------- A* Pathfinding Functions ---------------------------------------
# ----------------------------------------------------------------------------------------------------

def heuristic(a, b):
    """Manhattan distance heuristic function (considering only up/down/left/right movement)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_nearest_free_space(obstacle_grid, target_point, factory_w, factory_h, max_search_radius=5):
    """Finds the nearest free space from a given point using BFS."""
    if not (0 <= target_point[0] < factory_w and 0 <= target_point[1] < factory_h):
        return None
        
    if obstacle_grid[target_point[0]][target_point[1]] == 0:
        return target_point
        
    queue = deque([target_point])
    visited = set([target_point])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] 
    
    current_radius = 0
    while queue and current_radius <= max_search_radius:
        level_size = len(queue)
        current_radius += 1
        
        for _ in range(level_size):
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if not (0 <= nx < factory_w and 0 <= ny < factory_h):
                    continue
                    
                if (nx, ny) in visited:
                    continue
                    
                visited.add((nx, ny))
                
                if obstacle_grid[nx][ny] == 0:
                    return (nx, ny)
                    
                queue.append((nx, ny))
    
    return None

def find_machine_access_points(machine_pos_info, machine_def, obstacle_grid, factory_w, factory_h):
    """Finds the accessible points of the machine (adjacent free spaces)."""
    footprint_w, footprint_h = machine_def["footprint"]
    start_x, start_y = machine_pos_info["x"], machine_pos_info["y"]
    
    access_points = []
    
    # Check adjacent free spaces around the 4 sides
    for x in range(start_x, start_x + footprint_w):
        for dy in [-1, footprint_h]: # Top (y-1) and Bottom (y+height)
            y = start_y + dy
            if 0 <= x < factory_w and 0 <= y < factory_h and obstacle_grid[x][y] == 0:
                access_points.append((x, y))
    
    for y in range(start_y, start_y + footprint_h):
        for dx in [-1, footprint_w]: # Left (x-1) and Right (x+width)
            x = start_x + dx
            if 0 <= x < factory_w and 0 <= y < factory_h and obstacle_grid[x][y] == 0:
                access_points.append((x, y))
    
    # Remove duplicates from corners
    return list(set(access_points))

def get_best_access_point(machine_pos_info, machine_def, obstacle_grid, factory_w, factory_h):
    """Finds the optimal access point for a machine."""
    center_x = int(round(machine_pos_info["center_x"]))
    center_y = int(round(machine_pos_info["center_y"]))
    
    if (0 <= center_x < factory_w and 0 <= center_y < factory_h and 
        obstacle_grid[center_x][center_y] == 0):
        return (center_x, center_y)
    
    access_points = find_machine_access_points(machine_pos_info, machine_def, obstacle_grid, factory_w, factory_h)
    
    if access_points:
        center_point = (machine_pos_info["center_x"], machine_pos_info["center_y"])
        best_point = min(access_points, 
                          key=lambda p: abs(p[0] - center_point[0]) + abs(p[1] - center_point[1]))
        return best_point
    
    fallback_point = find_nearest_free_space(obstacle_grid, (center_x, center_y), 
                                             factory_w, factory_h, max_search_radius=10)
    
    if fallback_point:
        return fallback_point
    
    # Fallback to the closest grid point if all else fails
    return (center_x, center_y) 

def get_optimized_access_point_for_sequence(prev_access_point, current_pos_info, current_machine_def, 
                                            next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h):
    """Finds the optimal access point for the current machine, minimizing total Manhattan distance across three machines."""
    current_access_points = find_machine_access_points(current_pos_info, current_machine_def, 
                                                       obstacle_grid, factory_w, factory_h)
    
    if not current_access_points:
        return get_best_access_point(current_pos_info, current_machine_def, obstacle_grid, factory_w, factory_h)
    
    # Determine the best access point for the next machine (the exit point for the current segment)
    next_access_points = find_machine_access_points(next_pos_info, next_machine_def, 
                                                    obstacle_grid, factory_w, factory_h)
    
    if not next_access_points:
        next_target = get_best_access_point(next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h)
    else:
        next_center = (next_pos_info["center_x"], next_pos_info["center_y"])
        next_target = min(next_access_points, 
                          key=lambda p: abs(p[0] - next_center[0]) + abs(p[1] - next_center[1]))
    
    # Find the current access point that minimizes the total (prev -> current) + (current -> next) distance (Manhattan)
    best_point = None
    min_total_distance = float('inf')
    
    for current_point in current_access_points:
        dist_prev_to_current = abs(prev_access_point[0] - current_point[0]) + abs(prev_access_point[1] - current_point[1])
        dist_current_to_next = abs(current_point[0] - next_target[0]) + abs(current_point[1] - next_target[1])
        total_distance = dist_prev_to_current + dist_current_to_next
        
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_point = current_point
    
    if best_point is None:
        return get_best_access_point(current_pos_info, current_machine_def, obstacle_grid, factory_w, factory_h)
    
    return best_point

def a_star_search(grid_map, start_coords, goal_coords, factory_w, factory_h):
    """A* algorithm for shortest path on a grid (4 directions)."""
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Up/down/left/right movement

    close_set = set()
    came_from = {}
    gscore = {start_coords: 0}
    fscore = {start_coords: heuristic(start_coords, goal_coords)}
    oheap = [] 

    heapq.heappush(oheap, (fscore[start_coords], start_coords))
    
    while oheap:
        current_fscore, current_node = heapq.heappop(oheap)

        if current_node == goal_coords:
            path_data = []
            while current_node in came_from:
                path_data.append(current_node)
                current_node = came_from[current_node]
            path_data.append(start_coords) 
            return path_data[::-1] 

        close_set.add(current_node)
        for i, j in neighbors:
            neighbor = current_node[0] + i, current_node[1] + j
            
            if not (0 <= neighbor[0] < factory_w and 0 <= neighbor[1] < factory_h):
                continue
            if grid_map[neighbor[0]][neighbor[1]] == 1: 
                continue
                
            tentative_g_score = gscore[current_node] + 1 

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue
                
            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [item[1] for item in oheap]:
                came_from[neighbor] = current_node
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal_coords)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return None 

def visualize_layout_with_paths(layout_data, all_paths, flow_density_grid, filename_path="layout_with_paths.png", filename_heatmap="layout_congestion_heatmap.png"):
    factory_w = layout_data["factory_width"]
    factory_h = layout_data["factory_height"]
    machine_positions_map_str = layout_data["machine_positions_map"]
    machine_positions_map = {int(k): v for k, v in machine_positions_map_str.items()}
    process_sequence_list = layout_data["process_sequence"]
    machine_definitions_list = layout_data["machines_definitions"]

    num_total_machines_for_color = len(machine_definitions_list)
    cmap_machines = plt.colormaps.get_cmap('viridis')
    machines_dict_by_id = {m['id']: m for m in machine_definitions_list}

    # --- Visualization 1: Path Overlay ---
    fig_path, ax_path = plt.subplots(1, figsize=(factory_w/2.5, factory_h/2.8)) 
    ax_path.set_xlim(-0.5, factory_w - 0.5)
    ax_path.set_ylim(-0.5, factory_h - 0.5)
    ax_path.set_xticks(range(factory_w))
    ax_path.set_yticks(range(factory_h))
    ax_path.grid(True, linestyle='--', alpha=0.7)
    ax_path.set_aspect('equal', adjustable='box')

    # Draw Machines for Path Plot
    for machine_id, pos_data in machine_positions_map.items(): 
        machine_info = machines_dict_by_id.get(machine_id)
        if machine_info:
            x, y = pos_data['x'], pos_data['y']
            width, height = machine_info['footprint']
            clearance = machine_info.get('clearance', 0)
            
            normalized_id = machine_id / max(num_total_machines_for_color - 1, 1)
            face_color = cmap_machines(normalized_id)

            rect_body = patches.Rectangle((x - 0.5, y - 0.5), width, height,
                                          linewidth=1.5, edgecolor='black',
                                          facecolor=face_color, alpha=0.7)
            ax_path.add_patch(rect_body)
            ax_path.text(x + width/2 - 0.5, y + height/2 - 0.5, f"M{machine_id}",
                         ha='center', va='center', fontsize=7, color='white', weight='bold')

    # Draw Paths for Path Plot
    cmap_paths = plt.colormaps.get_cmap('cool') 
    num_paths = len(all_paths)
    path_number = 1
    labels_at_start_node = {}
    for i, path_segment in enumerate(all_paths):
        if path_segment:
            path_color = cmap_paths(i / max(num_paths - 1, 1))
            path_xs = [p[0] for p in path_segment]
            path_ys = [p[1] for p in path_segment]
            ax_path.plot(path_xs, path_ys, color=path_color, linewidth=2, alpha=0.8, marker='o', markersize=3)
            
            if len(path_segment) > 0:
                start_node_tuple = tuple(path_segment[0]) 

                offset_idx = labels_at_start_node.get(start_node_tuple, 0)
                adjusted_y_pos = path_segment[0][1] - 0.3 - (offset_idx * 1)
                adjusted_x_pos = path_segment[0][0] 

                ax_path.text(adjusted_x_pos, adjusted_y_pos, f"{path_number}",
                             color=path_color,
                             fontsize=10,
                             weight='bold',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', pad=0.2, boxstyle='round'))

                labels_at_start_node[start_node_tuple] = offset_idx + 1
                path_number += 1

    ax_path.set_title("Factory Layout with Optimized Continuous Paths", fontsize=14)
    ax_path.set_xlabel("Factory Width (X)")
    ax_path.set_ylabel("Factory Height (Y)")
    ax_path.invert_yaxis() 
    try:
        fig_path.savefig(filename_path, dpi=300, bbox_inches='tight')
        print(f"📊 Continuous Path Optimized Layout Visualization Image Saved: {filename_path}")
    except Exception as e:
        print(f"Error occurred while saving path visualization image: {e}")
    plt.close(fig_path)

    # --- Visualization 2: Congestion Heatmap --- (NEW)
    fig_heat, ax_heat = plt.subplots(1, figsize=(factory_w/2.5, factory_h/2.8)) 
    ax_heat.set_xlim(-0.5, factory_w - 0.5)
    ax_heat.set_ylim(-0.5, factory_h - 0.5)
    ax_heat.set_xticks(range(factory_w))
    ax_heat.set_yticks(range(factory_h))
    ax_heat.grid(True, linestyle='--', alpha=0.7)
    ax_heat.set_aspect('equal', adjustable='box')
    ax_heat.invert_yaxis() 

    # Prepare Flow Density Grid (Need to transpose for imshow)
    max_flow = max(max(row) for row in flow_density_grid) if flow_density_grid and any(any(row) for row in flow_density_grid) else 1
    # Note: flow_density_grid is (x, y), imshow expects (row, col) which maps to (y, x). 
    flow_density_grid_transposed = [list(row) for row in zip(*flow_density_grid)]

    # Draw Heatmap
    if max_flow > 0:
        # Custom colormap for flow (e.g., green to red, transparent where flow is 0)
        colors = [(0, 0, 0, 0), (0.1, 0.7, 0.1, 0.5), (1, 0.5, 0, 0.8), (1, 0, 0, 1)] # Transparent, Green, Orange, Red
        cmap_custom = LinearSegmentedColormap.from_list("flow_cmap", colors)
        
        im = ax_heat.imshow(flow_density_grid_transposed, cmap=cmap_custom, 
                            interpolation='nearest', origin='lower', extent=[-0.5, factory_w-0.5, -0.5, factory_h-0.5],
                            vmax=max_flow)
        
        # Add colorbar
        cbar = fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Material Flow Density (Path Overlaps)')

    # Draw Machines for Heatmap Plot
    for machine_id, pos_data in machine_positions_map.items(): 
        machine_info = machines_dict_by_id.get(machine_id)
        if machine_info:
            x, y = pos_data['x'], pos_data['y']
            width, height = machine_info['footprint']
            normalized_id = machine_id / max(num_total_machines_for_color - 1, 1)
            face_color = cmap_machines(normalized_id)

            rect_body = patches.Rectangle((x - 0.5, y - 0.5), width, height,
                                          linewidth=1.5, edgecolor='black',
                                          facecolor=face_color, alpha=0.7)
            ax_heat.add_patch(rect_body)
            ax_heat.text(x + width/2 - 0.5, y + height/2 - 0.5, f"M{machine_id}",
                         ha='center', va='center', fontsize=7, color='white', weight='bold')


    ax_heat.set_title("Factory Congestion Heatmap (A* Flow Analysis)", fontsize=14)
    ax_heat.set_xlabel("Factory Width (X)")
    ax_heat.set_ylabel("Factory Height (Y)")
    ax_heat.invert_yaxis() 
    try:
        fig_heat.savefig(filename_heatmap, dpi=300, bbox_inches='tight')
        print(f"📊 Congestion Heatmap Visualization Image Saved: {filename_heatmap}")
    except Exception as e:
        print(f"Error occurred while saving heatmap image: {e}")
    plt.close(fig_heat)

def generate_performance_summary(ga_metrics, a_star_metrics, process_sequence, machines_defs):
    """Prints a beautifully formatted summary table of all performance metrics."""
    
    # 1. Prepare Machine Utilization Data for the table
    # Keys are stored as strings in JSON, convert back to integers for lookup
    machine_util_data_str = ga_metrics.get("machine_utilization", {})
    machine_util_data = {int(k): v for k, v in machine_util_data_str.items()} 
    
    util_rows = []
    machines_dict = {m['id']: m for m in machines_defs}
    
    # 2. Find the Bottleneck Stage (highest utilization)
    bottleneck_mid = None
    max_util = 0.0
    
    for mid in process_sequence:
        name = machines_dict.get(mid, {}).get("name", "N/A")
        util = machine_util_data.get(mid, 0.0)
        cycle_time = machines_dict.get(mid, {}).get("cycle_time", 0)
        
        # Store for display
        util_rows.append((mid, name, cycle_time, util * 100))
        
        # Identify bottleneck (highest utilization)
        if util > max_util:
            max_util = util
            bottleneck_mid = mid

    # 3. Print Main Summary
    print("\n" + "="*80)
    print(f"{'FACTORY LAYOUT OPTIMIZATION SUMMARY':^80}")
    print("="*80)
    print(f"{'Metric':<35} | {'GA Value':<20} | {'A* Flow Value':<20}")
    print("-" * 80)
    print(f"{'Best Fitness Score (GA)':<35} | {ga_metrics.get('fitness', 0.0):<20.2f} | {'-':<20}")
    print(f"{'Hourly Throughput (TPH)':<35} | {ga_metrics.get('throughput', 0.0):<20.2f} | {'-':<20}")
    print(f"{'Target TPH':<35} | {TARGET_PRODUCTION_PER_HOUR:<20.0f} | {'-':<20}")
    print(f"{'Total Distance (Euclidean)':<35} | {ga_metrics.get('distance', 0.0):<20.2f} | {'-':<20}")
    print(f"{'Total Flow Distance (A*)':<35} | {'-':<20} | {a_star_metrics.get('total_a_star_distance', 0.0):<20.1f}")
    print(f"{'Total Travel Time (A* Flow)':<35} | {'-':<20} | {a_star_metrics.get('total_a_star_time_seconds', 0.0):<20.2f}")
    print(f"{'Area Utilization Ratio':<35} | {ga_metrics.get('utilization_ratio', 0.0)*100:<20.2f}% | {'-':<20}")
    print(f"{'Zone 1 Proximity Penalty':<35} | {ga_metrics.get('zone_penalty', 0.0):<20.2f} | {'-':<20}")
    print("-" * 80)
    
    # 4. Print Machine Utilization Rows
    print(f"\n{'MACHINE UTILIZATION ANALYSIS':^80}")
    print("-" * 80)
    print(f"{'ID':<4} | {'Name':<30} | {'Cycle Time (s)':<15} | {'Utilization Rate':<20}")
    print("-" * 80)
    
    for mid, name, cycle_time, util_rate in util_rows:
        # Highlight bottleneck based on highest util rate found
        highlight = " 🌟 BOTTLENECK" if mid == bottleneck_mid else ""
        print(f"M{mid:<3} | {name:<30} | {cycle_time:<15} | {util_rate:<19.2f}%{highlight}")
    print("="*80 + "\n")


def main_path_finder(layout_data_file=LAYOUT_DATA_FILE):
    """Loads GA results, calculates A* paths, performs congestion analysis, and visualizes them."""
    try:
        with open(layout_data_file, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
    except FileNotFoundError:
        print(f"\nError: Layout data file '{layout_data_file}' not found. Cannot run pathfinding.")
        return
    except json.JSONDecodeError:
        print(f"\nError: Layout data file '{layout_data_file}' has an invalid format. Cannot run pathfinding.")
        return

    factory_w = layout_data["factory_width"]
    factory_h = layout_data["factory_height"]
    machine_positions_str = layout_data["machine_positions_map"]
    machine_positions = {int(k): v for k, v in machine_positions_str.items()}
    process_sequence = layout_data["process_sequence"]
    machines_definitions = layout_data["machines_definitions"]
    machines_dict = {m['id']: m for m in machines_definitions}
    ga_metrics = layout_data["final_metrics"] # Retrieve GA metrics

    # 2. Create Grid Map for Pathfinding (Mark Obstacles)
    obstacle_grid = [[0 for _ in range(factory_h)] for _ in range(factory_w)]
    for machine_id_str, pos_info in machine_positions_str.items():
        machine_id = int(machine_id_str)
        m_def = machines_dict.get(machine_id)
        if m_def:
            footprint_w, footprint_h = m_def["footprint"]
            start_x, start_y = pos_info["x"], pos_info["y"]
            for i in range(start_x, start_x + footprint_w):
                for j in range(start_y, start_y + footprint_h):
                    if 0 <= i < factory_w and 0 <= j < factory_h:
                        obstacle_grid[i][j] = 1 # Machine body is an obstacle

    # 3. Pathfinding with Continuous Path Optimization
    print("\n🚀 Starting Single-Stroke Continuous Path Optimization...")
    all_paths_found = []
    access_points_cache = {} 
    
    # [NEW] Flow Density Grid for Congestion Analysis
    flow_density_grid = [[0 for _ in range(factory_h)] for _ in range(factory_w)]
    total_a_star_distance = 0
    total_a_star_time = 0

    first_machine_id = process_sequence[0]
    first_pos_info = machine_positions.get(first_machine_id)
    first_machine_def = machines_dict.get(first_machine_id)
    if first_pos_info and first_machine_def:
        access_points_cache[first_machine_id] = get_best_access_point(
            first_pos_info, first_machine_def, obstacle_grid, factory_w, factory_h)
    
    # Path report header
    print("\n--- A* Path Segment Analysis ---")
    print(f"{'Segment':<15} | {'Start Node':<12} | {'Goal Node':<12} | {'A* Distance':<15} | {'Travel Time (s)':<15}")
    print("-" * 70)

    for i in range(len(process_sequence) - 1):
        current_machine_id = process_sequence[i]
        next_machine_id = process_sequence[i+1]

        current_pos_info = machine_positions.get(current_machine_id)
        next_pos_info = machine_positions.get(next_machine_id)
        current_machine_def = machines_dict.get(current_machine_id)
        next_machine_def = machines_dict.get(next_machine_id)

        if not current_pos_info or not next_pos_info or not current_machine_def or not next_machine_def:
            all_paths_found.append(None); continue

        start_node = access_points_cache.get(current_machine_id)
        if start_node is None:
            start_node = get_best_access_point(current_pos_info, current_machine_def, obstacle_grid, factory_w, factory_h)
            access_points_cache[current_machine_id] = start_node
        
        # Continuous Optimization for intermediate nodes
        if i < len(process_sequence) - 2: 
            next_next_machine_id = process_sequence[i+2]
            next_next_pos_info = machine_positions.get(next_next_machine_id)
            next_next_machine_def = machines_dict.get(next_next_machine_id)
            
            if next_next_pos_info and next_next_machine_def:
                goal_node = get_optimized_access_point_for_sequence(
                    start_node, next_pos_info, next_machine_def,
                    next_next_pos_info, next_next_machine_def,
                    obstacle_grid, factory_w, factory_h)
                access_points_cache[next_machine_id] = goal_node
            else:
                goal_node = get_best_access_point(next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h)
                access_points_cache[next_machine_id] = goal_node
        else: # Last segment
            goal_node = get_best_access_point(next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h)
            access_points_cache[next_machine_id] = goal_node
        
        path = a_star_search(obstacle_grid, start_node, goal_node, factory_w, factory_h)
        a_star_dist = 0
        travel_time = 0
        
        if path:
            all_paths_found.append(path)
            a_star_dist = len(path) - 1 # Path length = number of steps
            travel_time = a_star_dist / MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND
            
            # [NEW] Congestion Analysis: Update flow density grid
            for x, y in path:
                if 0 <= x < factory_w and 0 <= y < factory_h and obstacle_grid[x][y] == 0:
                    flow_density_grid[x][y] += 1
        else:
            all_paths_found.append([start_node, goal_node]) # Fallback for no path found
            # Use Manhattan as approximation for distance if A* fails
            a_star_dist = abs(start_node[0] - goal_node[0]) + abs(start_node[1] - goal_node[1]) 
            travel_time = a_star_dist / MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND
            
        total_a_star_distance += a_star_dist
        total_a_star_time += travel_time
        
        print(f"M{current_machine_id} -> M{next_machine_id:<7} | {str(start_node):<12} | {str(goal_node):<12} | {a_star_dist:<15.1f} | {travel_time:<15.2f}")

    print("-" * 70)
    # Corrected f-string formatting
    print(f"{'A* Totals':<30} | {'':<12} | {'Total A* Distance':<15} | {total_a_star_distance:<15.1f}")
    print(f"{'':<30} | {'':<12} | {'Total Flow Time (s)':<15} | {total_a_star_time:<15.2f}")
    
    # Package A* metrics
    a_star_metrics = {
        "total_a_star_distance": total_a_star_distance,
        "total_a_star_time_seconds": total_a_star_time
    }

    # Save the updated A* metrics back into the JSON file
    if "final_metrics" in layout_data and layout_data["final_metrics"]:
        layout_data["final_metrics"].update(a_star_metrics)
        
        try:
            with open(layout_data_file, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=4)
            print(f"💾 Updated layout metrics (A* distance, flow time) saved to {layout_data_file}.")
        except Exception as e:
            print(f"Error saving updated layout data to JSON: {e}")

    # 4. Visualize Paths & Congestion
    print("🎨 Visualizing Single-Stroke Optimization Result and Congestion Heatmap...")
    visualize_layout_with_paths(layout_data, all_paths_found, flow_density_grid)
    
    # 5. Generate Final Summary Table (NEW)
    generate_performance_summary(ga_metrics, a_star_metrics, process_sequence, machines_definitions)


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------- COMBINED MAIN --------------------------------------------
# ----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    print("--- Factory Layout Optimization (Genetic Algorithm + Pathfinding) ---")
    signal.signal(signal.SIGINT, signal_handler)

    machines_for_ga_processing_order = [next(m for m in machines_definitions if m["id"] == pid) for pid in PROCESS_SEQUENCE]
    
    print(f"Initial population generation in progress (Size: {POPULATION_SIZE})...")
    # THE CALL TO create_individual IS NOW DEFINED ABOVE
    population = [create_individual(machines_for_ga_processing_order, FACTORY_WIDTH, FACTORY_HEIGHT) for _ in range(POPULATION_SIZE)]
    print("Initial population generation complete.")
    
    best_overall_fitness = -float('inf')
    best_overall_chromosome = None
    # Initialize best_overall_metrics here, to ensure it's a dict that can store the utilization data
    best_overall_metrics = {} 
    
    generation_best_fitness_log = []
    generation_avg_fitness_log = []
    generation_best_distance_log = []
    generation_best_throughput_log = []
    generation_valid_ratio_log = []
    generation_best_zone_penalty_log = []
    generation_best_utilization_log = []

    for generation in range(1, NUM_GENERATIONS + 1):
        if interrupted:
            print(f"\n🛑 Genetic Algorithm termination requested (Generation {generation})")
            break

        all_eval_results = []
        current_gen_total_fitness = 0
        current_gen_valid_individuals_count = 0

        for chromo in population:
            eval_dict = calculate_fitness(chromo, machines_for_ga_processing_order, PROCESS_SEQUENCE,
                                          FACTORY_WIDTH, FACTORY_HEIGHT, TARGET_PRODUCTION_PER_HOUR,
                                          MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND)
            all_eval_results.append((chromo, eval_dict))
            if eval_dict["is_valid"]:
                current_gen_total_fitness += eval_dict["fitness"]
                current_gen_valid_individuals_count += 1
        
        if not all_eval_results:
            print(f"Generation {generation}: No evaluation results! Algorithm halted."); break

        current_gen_best_item = max(all_eval_results, key=lambda item: item[1]['fitness'])
        current_gen_best_chromosome = current_gen_best_item[0]
        current_gen_best_eval_dict = current_gen_best_item[1]
        
        current_gen_best_fitness = current_gen_best_eval_dict['fitness']
        current_gen_best_distance = current_gen_best_eval_dict['distance']
        current_gen_best_throughput = current_gen_best_eval_dict['throughput']
        
        generation_best_fitness_log.append(current_gen_best_fitness)
        generation_best_distance_log.append(current_gen_best_distance) 
        generation_best_throughput_log.append(current_gen_best_throughput) 
        generation_best_zone_penalty_log.append(current_gen_best_eval_dict.get('zone_penalty', 0.0))
        generation_best_utilization_log.append(current_gen_best_eval_dict.get('utilization_ratio', 0.0))

        valid_ratio = (current_gen_valid_individuals_count / POPULATION_SIZE) * 100 if POPULATION_SIZE > 0 else 0
        generation_valid_ratio_log.append(valid_ratio)
        
        current_gen_avg_fitness = (current_gen_total_fitness / current_gen_valid_individuals_count) if current_gen_valid_individuals_count > 0 else -float('inf')
        generation_avg_fitness_log.append(current_gen_avg_fitness)

        if current_gen_best_fitness > best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_chromosome = current_gen_best_chromosome
            # FIX: Deep copy the entire evaluation dictionary, including 'machine_utilization'
            best_overall_metrics = copy.deepcopy(current_gen_best_eval_dict) 
            print(f" 🌟 Generation {generation}: New best fitness found! {best_overall_fitness:.2f} (Dist: {current_gen_best_distance:.2f}, TPH: {current_gen_best_throughput:.2f}, ZonePen: {best_overall_metrics['zone_penalty']:.2f})")

        print(f"Generation {generation}/{NUM_GENERATIONS} - BestF: {current_gen_best_fitness:.2f}, AvgF: {current_gen_avg_fitness:.2f}, ValidRatio: {valid_ratio:.1f}%")
        
        new_population = []
        all_eval_results.sort(key=lambda item: item[1]['fitness'], reverse=True)
        for i in range(ELITISM_COUNT):
            if i < len(all_eval_results) and all_eval_results[i][1]["is_valid"]:
                new_population.append(all_eval_results[i][0])

        num_offspring_to_generate = POPULATION_SIZE - len(new_population)
        eligible_parents_for_selection = [item for item in all_eval_results if item[1]["is_valid"]]
        if not eligible_parents_for_selection: eligible_parents_for_selection = all_eval_results 

        current_offspring_count = 0
        while current_offspring_count < num_offspring_to_generate:
            if not eligible_parents_for_selection: break
            parent1_chromo = selection(eligible_parents_for_selection, TOURNAMENT_SIZE)
            parent2_chromo = selection(eligible_parents_for_selection, TOURNAMENT_SIZE)
            
            if parent1_chromo is None or parent2_chromo is None: 
                # Fallback to random choice if tournament selection fails (e.g., small, uniform population)
                if eligible_parents_for_selection:
                    if parent1_chromo is None: parent1_chromo = random.choice(eligible_parents_for_selection)[0]
                    if parent2_chromo is None: parent2_chromo = random.choice(eligible_parents_for_selection)[0]
                else: 
                    break

            child1_chromo, child2_chromo = crossover(parent1_chromo, parent2_chromo, CROSSOVER_RATE)
            if random.random() < MUTATION_RATE: child1_chromo = mutate(child1_chromo, machines_for_ga_processing_order, FACTORY_WIDTH, FACTORY_HEIGHT, mutation_rate_per_gene=0.05)
            if random.random() < MUTATION_RATE: child2_chromo = mutate(child2_chromo, machines_for_ga_processing_order, FACTORY_WIDTH, FACTORY_HEIGHT, mutation_rate_per_gene=0.05)
            new_population.append(child1_chromo); current_offspring_count +=1
            if current_offspring_count < num_offspring_to_generate: new_population.append(child2_chromo); current_offspring_count +=1
        
        while len(new_population) < POPULATION_SIZE:
            new_population.append(create_individual(machines_for_ga_processing_order, FACTORY_WIDTH, FACTORY_HEIGHT))
        population = new_population[:POPULATION_SIZE]

    # --- Final Result Output & Data Export ---
    print("\n--- Final Results (Genetic Algorithm) ---")
    print(f"Process Sequence to be Used: {PROCESS_SEQUENCE}")
    
    if best_overall_chromosome and best_overall_metrics.get("is_valid"):
        print(f"Best Fitness: {best_overall_fitness:.2f}")
        final_grid_layout = initialize_layout_grid(FACTORY_WIDTH, FACTORY_HEIGHT)
        final_machine_positions_map = {}

        # Recalculate positions based on best chromosome
        for i, machine_def_item in enumerate(machines_for_ga_processing_order):
            pos_x_final, pos_y_final = best_overall_chromosome[i]
            place_machine_on_grid(final_grid_layout, machine_def_item["id"], machine_def_item["footprint"], pos_x_final, pos_y_final)
            final_machine_positions_map[machine_def_item["id"]] = {
                "x": pos_x_final, "y": pos_y_final,
                "center_x": pos_x_final + machine_def_item["footprint"][0] / 2.0,
                "center_y": pos_y_final + machine_def_item["footprint"][1] / 2.0,
            }
        
        print_layout(final_grid_layout, final_machine_positions_map)
        
        layout_image_filename = 'ga_optimized_layout_visualization.png'
        if interrupted: layout_image_filename = 'ga_optimized_layout_visualization_interrupted.png'
        visualize_layout_plt(final_grid_layout, 
                             final_machine_positions_map, 
                             FACTORY_WIDTH, FACTORY_HEIGHT,
                             PROCESS_SEQUENCE,
                             machines_definitions,
                             filename=layout_image_filename)

        print("\n--- Final Performance Metrics ---")
        print(f"Total travel distance (Euclidean): {best_overall_metrics['distance']:.2f}")
        print(f"Hourly throughput: {best_overall_metrics['throughput']:.2f} (Target: {TARGET_PRODUCTION_PER_HOUR})")
        print(f"Zone 1 Proximity Penalty: {best_overall_metrics['zone_penalty']:.2f} (Target Max Spread: {ZONE_1_MAX_SPREAD_DISTANCE})")
        print(f"MHS Turn Penalty Component: {best_overall_metrics['mhs_turn_penalty']:.3f}")
        print(f"Area Utilization Ratio: {best_overall_metrics['utilization_ratio']:.2%} (Target Min: {AREA_UTIL_MIN_THRESHOLD:.0%})")
        

        # --- Export Data for Pathfinding ---
        # Note: Need to convert keys to string for JSON saving
        final_machine_positions_map_str_keys = {str(k): v for k, v in final_machine_positions_map.items()}
        save_layout_data(final_machine_positions_map_str_keys, PROCESS_SEQUENCE, machines_definitions, FACTORY_WIDTH, FACTORY_HEIGHT, best_overall_metrics, LAYOUT_DATA_FILE)

    else: print("Did not find a valid optimal layout.")

    # --- Run Pathfinding Analysis ---
    if os.path.exists(LAYOUT_DATA_FILE):
        main_path_finder(LAYOUT_DATA_FILE)
    else:
        print("⚠️ Skipping pathfinding analysis as layout data file was not successfully created.")

    # --- Graph Generation (Updated to 6 plots) ---
    try:
        plt.figure(figsize=(15, 10)) # Increased size for 6 plots
        
        # 1. Fitness Progress
        plt.subplot(2, 3, 1)
        plt.plot(generation_best_fitness_log, label='Best Fitness', color='blue')
        plt.plot(generation_avg_fitness_log, label='Average Fitness (Valid)', color='cyan', linestyle='--')
        plt.xlabel('Generation'); plt.ylabel('Fitness'); plt.title('GA Fitness Progress'); plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.7)

        # 2. Total Distance
        plt.subplot(2, 3, 2)
        plot_distances = [d if d != float('inf') else max(filter(lambda x: x!=float('inf'), generation_best_distance_log), default=1000)*1.1 for d in generation_best_distance_log]
        if not any(d != float('inf') for d in generation_best_distance_log) and generation_best_distance_log: plot_distances = [0] * len(generation_best_distance_log) 
        plt.plot(plot_distances, label='Best Individual Euclidean Distance', color='green')
        plt.xlabel('Generation'); plt.ylabel('Total Distance'); plt.title('Best Individual Distance'); plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.7)
        if any(d == float('inf') for d in generation_best_distance_log): plt.text(0.05, 0.95, "Note: 'inf' distances capped", transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')

        # 3. Throughput vs. Target
        plt.subplot(2, 3, 3)
        plt.plot(generation_best_throughput_log, label='Best Individual Throughput', color='red')
        plt.xlabel('Generation'); plt.ylabel('Throughput')
        plt.axhline(y=TARGET_PRODUCTION_PER_HOUR, color='gray', linestyle=':', label=f'Target TPH ({TARGET_PRODUCTION_PER_HOUR})')
        plt.title('Best Individual Throughput'); plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.7)

        # 4. Valid Individuals Ratio
        plt.subplot(2, 3, 4)
        plt.plot(generation_valid_ratio_log, label='Valid Individuals Ratio (%)', color='purple')
        plt.xlabel('Generation'); plt.ylabel('Valid Ratio (%)'); plt.ylim(0, 105)
        plt.title('Valid Individuals Ratio'); plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.7)

        # 5. Zone Constraint Penalty (NEW)
        plt.subplot(2, 3, 5)
        plt.plot(generation_best_zone_penalty_log, label='Zone Constraint Penalty (Best)', color='orange')
        plt.xlabel('Generation'); plt.ylabel('Penalty Value'); 
        plt.title('Zone Proximity Penalty'); plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.7)

        # 6. Area Utilization Ratio (NEW)
        plt.subplot(2, 3, 6)
        plt.plot([r * 100 for r in generation_best_utilization_log], label='Area Utilization (Best)', color='brown')
        plt.xlabel('Generation'); plt.ylabel('Utilization (%)'); 
        plt.axhline(y=AREA_UTIL_MIN_THRESHOLD * 100, color='gray', linestyle=':', label=f'Target Min ({AREA_UTIL_MIN_THRESHOLD*100:.0f}%)')
        plt.title('Area Utilization Ratio'); plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        graph_filename_output = 'ga_factory_layout_analysis_plots.png'
        if interrupted: graph_filename_output = 'ga_factory_layout_analysis_plots_interrupted.png'
        plt.savefig(graph_filename_output, dpi=300)
        print(f"📊 Analysis Graph Save Complete: {graph_filename_output}")
        plt.close()
    except ImportError: print("The matplotlib library is not installed, so graphs cannot be generated.")
    except Exception as e_graph: print(f"Unexpected error occurred during graph generation: {e_graph}")
        
    if interrupted: print("\n✅ Safely terminated.")
    print("\nProgram terminated.")

ADD MORE INNOVATIONS TO THIS CODE, I NEED TO HAVE MORE VERY USEFUL OUTPUT CONTENT DO SOMETHING AND HELP ME WITH IT
