import streamlit as st
import random
import math
import copy
import os
import json
import heapq
import io
from contextlib import redirect_stdout
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Factory Layout Optimizer")

# --- CUSTOM CSS ("Dark Purple / Orange" Theme) ---
CSS_STYLE = """
<style>

/* --- Main App Background (Dark Purple) --- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #eda1d9 0%, #FFFFFF 100%);
    background-attachment: fixed;
    color: #FFFFFF; /* Light purple text for main body */
}

/* --- Sidebar Styling (Light Purple) --- */
[data-testid="stSidebar"] {
    background-color: #FFFFFF; /* Light Purple */
    border-right: 2px solid #4F359B;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: ##2E1A47F; /* Dark text in sidebar for contrast */
}
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
    color: ##2E1A47; /* Sidebar input labels */
}


/* --- Main "Run" Button (Orange) --- */
[data-testid="stButton"] button {
    background: linear-gradient(90deg, #FF8C00, #FFA500); /* Orange gradient */
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
}
[data-testid="stButton"] button:hover {
    transform: scale(1.05); /* "Pop" effect */
    box-shadow: 0 6px 20px rgba(255, 140, 0, 0.5);
}
[data-testid="stButton"] button:active {
    transform: scale(0.98);
}

/* --- Headings --- */
h1, h2, h3 {
    color: #D8CCFF; /* Light purple for headings */
}
h1 {
    color: #FFFFFF; /* White for main title */
}

/* --- Sidebar Expander Headers (Darker text on light bg) --- */
[data-testid="stExpander"] summary {
    background-color: #D8CCFF; /* Slightly darker light purple */
    color: #2E1A47; /* Dark purple text */
    border-radius: 8px;
    transition: all 0.2s ease;
}
[data-testid="stExpander"] summary:hover {
    background-color: #C8B8FF;
}

/* --- Metric Cards (Interactive) --- */
[data-testid="stMetric"] {
    background-color: #3E2C5E; /* Medium-dark purple */
    border: 1px solid #FF8C00; /* Orange border */
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 16px rgba(255, 140, 0, 0.15);
}
/* Metric label and value colors */
[data-testid="stMetric"] label {
    color: #D8CCFF; /* Light purple for label */
}
[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #FFFFFF; /* White for value */
}
[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    color: #90EE90; /* Light green for positive delta */
}
[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {
    fill: #90EE90;
}

/* --- Tab Styling (Orange accent) --- */
[data-testid="stTabs"] [data-baseweb="tab-list"] button {
    color: #D8CCFF; /* Inactive tab text */
}
[data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #3E2C5E;
    color: #FFFFFF; /* Active tab text */
    border-radius: 8px 8px 0 0;
    border-bottom: 3px solid #FFA500; /* Active tab underline (Orange) */
}

/* --- Code Block Styling --- */
[data-testid="stCodeBlock"] {
    background-color: #1F1030;
    border: 1px solid #4F359B;
    border-radius: 8px;
}

/* --- Info/Success Box --- */
[data-testid="stInfo"] {
    background-color: rgba(230, 224, 255, 0.1);
    border: 1px solid #D8CCFF;
    color: #E6E0FF;
}
[data-testid="stSuccess"] {
    background-color: rgba(144, 238, 144, 0.1);
    border: 1px solid #90EE90;
    color: #E6E0FF;
}

</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- End of CSS ---


st.title("🏭 GA Factory Layout & A* Flow Optimizer")
st.info(
    "Configure your factory, machines, and optimization parameters in the sidebar. "
    "Click 'Run Optimization' to start the Genetic Algorithm and A* pathfinding analysis."
)

# --- Original Code (Refactored into Functions) ---

# Note: The Machine class is not strictly used by the GA logic, 
# as it operates on dictionaries, but it's good practice to keep.
class Machine:
    def __init__(self, id, name, footprint, cycle_time=0, clearance=0, wall_affinity=False):
        self.id = id; self.name = name; self.footprint = footprint; self.cycle_time = cycle_time
        self.clearance = clearance; self.wall_affinity = wall_affinity; self.position = None

# Default Machine Definitions (as a JSON string for the text area)
DEFAULT_MACHINES_JSON = """
[
    {"id": 0, "name": "Raw Material Input", "footprint": [2, 2], "cycle_time": 20, "clearance": 1, "zone_group": null},
    {"id": 1, "name": "1st Cutting", "footprint": [3, 3], "cycle_time": 35, "clearance": 1, "zone_group": 1},
    {"id": 2, "name": "Milling Process", "footprint": [4, 2], "cycle_time": 45, "clearance": 1, "zone_group": 1},
    {"id": 3, "name": "Drilling", "footprint": [2, 2], "cycle_time": 25, "clearance": 1, "zone_group": 1},
    {"id": 4, "name": "Heat Treatment A", "footprint": [3, 4], "cycle_time": 70, "clearance": 2, "zone_group": null},
    {"id": 5, "name": "Precision Machining A", "footprint": [3, 2], "cycle_time": 40, "clearance": 1, "zone_group": 2},
    {"id": 6, "name": "Assembly A", "footprint": [2, 3], "cycle_time": 55, "clearance": 2, "zone_group": 3},
    {"id": 7, "name": "Final Inspection A", "footprint": [1, 2], "cycle_time": 15, "clearance": 1, "zone_group": 3},
    {"id": 8, "name": "2nd Cutting", "footprint": [3, 2], "cycle_time": 30, "clearance": 1, "zone_group": 1},
    {"id": 9, "name": "Surface Treatment", "footprint": [2, 4], "cycle_time": 50, "clearance": 2, "zone_group": null},
    {"id": 10, "name": "Washing Process 1", "footprint": [2, 2], "cycle_time": 20, "clearance": 1, "zone_group": 2},
    {"id": 11, "name": "Heat Treatment B", "footprint": [4, 4], "cycle_time": 75, "clearance": 2, "zone_group": null},
    {"id": 12, "name": "Precision Machining B", "footprint": [2, 3], "cycle_time": 42, "clearance": 1, "zone_group": 2},
    {"id": 13, "name": "Component Assembly", "footprint": [3, 3], "cycle_time": 60, "clearance": 1, "zone_group": 3},
    {"id": 14, "name": "Quality Inspection B", "footprint": [2, 1], "cycle_time": 18, "clearance": 1, "zone_group": 3},
    {"id": 15, "name": "Packaging Line A", "footprint": [4, 3], "cycle_time": 30, "clearance": 2, "zone_group": null}
]
"""
# Default Process Sequence (as a JSON string)
DEFAULT_PROCESS_SEQUENCE_JSON = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"

# Default Zone 1 Targets (as a JSON string)
DEFAULT_ZONE_1_TARGETS_JSON = "[1, 2, 3, 8]"

# Time Constants
SECONDS_PER_HOUR = 3600

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
    return abs(pos1['center_x'] - pos2['center_x']) + abs(pos1['center_y'] - pos2['center_y'])

def calculate_total_distance(machine_positions, process_sequence, distance_type='euclidean'):
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
    total_footprint_area = 0
    
    for machine_def in machines_defs_ordered_by_proc_seq:
        if machine_def["id"] in machine_positions:
            w, h = machine_def["footprint"]
            total_footprint_area += w * h
            
    factory_area = factory_w * factory_h
    utilization_ratio = total_footprint_area / factory_area
    
    return total_footprint_area, utilization_ratio

def calculate_zone_penalty(machine_positions, constraint_params):
    zone_target_ids = constraint_params['zone_1_target_machines']
    max_spread_dist = constraint_params['zone_1_max_spread_distance']

    zone_machines_pos = [machine_positions[m_id] for m_id in zone_target_ids if m_id in machine_positions]
    
    if len(zone_machines_pos) < 2: return 0.0
    
    max_dist = 0.0
    for i in range(len(zone_machines_pos)):
        for j in range(i + 1, len(zone_machines_pos)):
            pos1 = zone_machines_pos[i]
            pos2 = zone_machines_pos[j]
            dist = math.sqrt((pos1['center_x'] - pos2['center_x'])**2 + (pos1['center_y'] - pos2['center_y'])**2)
            if dist > max_dist: max_dist = dist
            
    if max_dist > max_spread_dist:
        penalty = (max_dist - max_spread_dist) ** 2
        return penalty
    return 0.0

def calculate_machine_utilization_and_bottleneck(machine_positions, process_sequence, all_machines_data, travel_speed):
    if not machine_positions or not process_sequence: return (0.0, {})
    
    stage_times = {}
    
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

        current_stage_total_time = machine_cycle_time + travel_time_to_next
        stage_times[current_machine_id] = current_stage_total_time

    max_stage_time = max(stage_times.values()) if stage_times else 0.0
    if max_stage_time <= 0 or max_stage_time == float('inf'):
        return (0.0, {})

    utilization_data = {}
    for i, machine_id in enumerate(process_sequence):
        machine_cycle_time = get_machine_cycle_time(machine_id, all_machines_data)
        
        if machine_cycle_time == float('inf') or max_stage_time == 0.0:
            utilization_data[machine_id] = 0.0
        else:
            utilization = machine_cycle_time / max_stage_time
            utilization_data[machine_id] = min(1.0, utilization)
            
    return (max_stage_time, utilization_data)


def calculate_fitness(chromosome, machines_defs_ordered_by_proc_seq, process_seq_ids,
                      factory_w, factory_h, target_prod_throughput, material_travel_speed,
                      all_machines_data, fitness_weights, constraint_params):
    
    grid = initialize_layout_grid(factory_w, factory_h)
    machine_positions = {}
    all_machines_placed_successfully = True
    
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
        machine_positions, process_seq_ids, all_machines_data, material_travel_speed)
    
    result["machine_utilization"] = utilization_data
    throughput = SECONDS_PER_HOUR / max_stage_time if max_stage_time > 0 and max_stage_time != float('inf') else 0.0
    
    if max_stage_time == float('inf') or throughput == 0.0: return result

    # 2. Euclidean Distance
    total_euclidean_dist = calculate_total_distance(machine_positions, process_seq_ids, distance_type='euclidean')

    # 3. Zone Constraint Penalty (NEW)
    zone_penalty = calculate_zone_penalty(machine_positions, constraint_params)
    result["zone_penalty"] = zone_penalty
    
    # 4. Area Utilization Bonus (NEW)
    _, utilization_ratio = calculate_area_metrics(machine_positions, machines_defs_ordered_by_proc_seq, factory_w, factory_h)
    result["utilization_ratio"] = utilization_ratio
    utilization_bonus = 0.0
    if utilization_ratio >= constraint_params['area_util_min_threshold']:
        utilization_bonus = (utilization_ratio - constraint_params['area_util_min_threshold']) * factory_w * factory_h * fitness_weights["utilization_bonus"]

    # 5. MHS Turn Penalty (Approximation using Manhattan distance) (NEW)
    total_manhattan_dist = calculate_total_distance(machine_positions, process_seq_ids, distance_type='manhattan')
    mhs_turn_penalty_value = total_manhattan_dist * fitness_weights["mhs_turn_penalty"]
    result["mhs_turn_penalty"] = mhs_turn_penalty_value

    # Combine fitness components
    fitness_val = (fitness_weights["throughput"] * throughput) \
                  - (fitness_weights["distance"] * total_euclidean_dist) \
                  - (fitness_weights["zone_penalty"] * zone_penalty) \
                  - mhs_turn_penalty_value \
                  + utilization_bonus
                  
    if throughput >= target_prod_throughput: 
        fitness_val += throughput * fitness_weights["bonus_for_target_achievement"]
    
    result["fitness"] = fitness_val
    result["distance"] = total_euclidean_dist
    result["throughput"] = throughput
    result["is_valid"] = True
    return result


def print_layout(grid, machine_positions, factory_w, factory_h, process_sequence, machines_definitions):
    """This function is designed to be captured by redirect_stdout."""
    print("--- Current Layout ---")
    for row_idx_actual in range(factory_h -1, -1, -1):
        row_to_print = [f"{grid[col_idx_actual][row_idx_actual]:2d}" if grid[col_idx_actual][row_idx_actual] != -1 else "__" for col_idx_actual in range(factory_w)]
        print(f"Y{row_idx_actual:<2}| " + " ".join(row_to_print))
    header = "    " + " ".join(f"X{i:<2}" for i in range(factory_w))
    print("-" * len(header)); print(header)
    if not machine_positions: print("No facilities placed"); return
    print("\n--- Machine Positions (Top-Left, Center) ---")
    for machine_id in process_sequence:
        if machine_id in machine_positions:
            pos_data = machine_positions[machine_id]
            machine_def = next((m for m in machines_definitions if m["id"] == machine_id), None)
            machine_name = machine_def["name"] if machine_def else "Unknown"
            print(f"Machine ID {machine_id} ({machine_name}): Top-Left ({pos_data['x']}, {pos_data['y']}), Center ({pos_data['center_x']:.1f}, {pos_data['center_y']:.1f})")

def visualize_layout_plt(grid_layout_to_show, machine_positions_map, factory_w, factory_h, 
                         process_sequence_list, machine_definitions_list, constraint_params):
    """Visualizes the final layout using matplotlib. Returns a fig object."""
    
    fig, ax = plt.subplots(1, figsize=(max(10, factory_w/2), max(10, factory_h/2 + 1))) 
    
    # Style for dark mode
    fig.patch.set_facecolor('#4F359B') # Main BG
    ax.set_facecolor('#3E2C5E') # Plot BG
    ax.tick_params(colors='#D8CCFF')
    ax.xaxis.label.set_color('#D8CCFF')
    ax.yaxis.label.set_color('#D8CCFF')
    ax.title.set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#4F359B')
    ax.spines['top'].set_color('#4F359B')
    ax.spines['left'].set_color('#4F359B')
    ax.spines['right'].set_color('#4F359B')

    ax.set_xlim(-0.5, factory_w - 0.5)
    ax.set_ylim(-0.5, factory_h - 0.5)
    ax.set_xticks(range(factory_w))
    ax.set_yticks(range(factory_h))
    ax.set_xticklabels(range(factory_w))
    ax.set_yticklabels(range(factory_h))
    ax.grid(True, linestyle='--', alpha=0.3, color='#FFA500') # Orange grid
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() 

    cmap = plt.colormaps.get_cmap('viridis')
    num_machines = len(machine_definitions_list)
    machines_dict_by_id = {m['id']: m for m in machine_definitions_list}

    # --- Draw Zone Constraint Boundary (NEW) ---
    zone_1_target_machines = constraint_params['zone_1_target_machines']
    zone_1_max_spread_distance = constraint_params['zone_1_max_spread_distance']
    
    zone_1_centers = [machine_positions_map[m_id] for m_id in zone_1_target_machines if m_id in machine_positions_map]
    if len(zone_1_centers) > 0:
        center_x_coords = [c['center_x'] for c in zone_1_centers]
        center_y_coords = [c['center_y'] for c in zone_1_centers]
        
        min_cx, max_cx = min(center_x_coords), max(center_x_coords)
        min_cy, max_cy = min(center_y_coords), max(center_y_coords)
        
        avg_cx, avg_cy = (min_cx + max_cx) / 2, (min_cy + max_cy) / 2
        
        max_dist_from_avg_sq = 0
        for cx, cy in zip(center_x_coords, center_y_coords):
            dist_sq = (cx - avg_cx)**2 + (cy - avg_cy)**2
            if dist_sq > max_dist_from_avg_sq:
                max_dist_from_avg_sq = dist_sq
        
        required_radius = zone_1_max_spread_distance / 2.0
        
        circle_required = patches.Circle((avg_cx - 0.5, avg_cy - 0.5), 
                                         radius=required_radius, 
                                         linewidth=2, 
                                         edgecolor='red', 
                                         facecolor='none', 
                                         linestyle='--', 
                                         alpha=0.6,
                                         label=f"Zone 1 Max Spread ({zone_1_max_spread_distance})")
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
                                              facecolor=cmap(color_value), alpha=0.8)
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
    legend = plt.legend()
    plt.setp(legend.get_texts(), color='#D8CCFF')
    
    return fig

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

# ----------------------------------------------------------------------------------------------------
# ----------------------------------- A* Pathfinding Functions ---------------------------------------
# ----------------------------------------------------------------------------------------------------

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_nearest_free_space(obstacle_grid, target_point, factory_w, factory_h, max_search_radius=5):
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
    footprint_w, footprint_h = machine_def["footprint"]
    start_x, start_y = machine_pos_info["x"], machine_pos_info["y"]
    
    access_points = []
    
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
    
    return list(set(access_points))

def get_best_access_point(machine_pos_info, machine_def, obstacle_grid, factory_w, factory_h):
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
    
    return (center_x, center_y) 

def get_optimized_access_point_for_sequence(prev_access_point, current_pos_info, current_machine_def, 
                                            next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h):
    current_access_points = find_machine_access_points(current_pos_info, current_machine_def, 
                                                       obstacle_grid, factory_w, factory_h)
    
    if not current_access_points:
        return get_best_access_point(current_pos_info, current_machine_def, obstacle_grid, factory_w, factory_h)
    
    next_access_points = find_machine_access_points(next_pos_info, next_machine_def, 
                                                    obstacle_grid, factory_w, factory_h)
    
    if not next_access_points:
        next_target = get_best_access_point(next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h)
    else:
        next_center = (next_pos_info["center_x"], next_pos_info["center_y"])
        next_target = min(next_access_points, 
                          key=lambda p: abs(p[0] - next_center[0]) + abs(p[1] - next_center[1]))
    
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
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

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

def visualize_layout_with_paths(layout_data, all_paths, flow_density_grid):
    """Visualizes paths and heatmap. Returns two fig objects."""
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
    fig_path, ax_path = plt.subplots(1, figsize=(max(10, factory_w/2.5), max(10, factory_h/2.8))) 
    
    # Style for dark mode
    fig_path.patch.set_facecolor('#4F359B') # Main BG
    ax_path.set_facecolor('#3E2C5E') # Plot BG
    ax_path.tick_params(colors='#D8CCFF')
    ax_path.xaxis.label.set_color('#D8CCFF')
    ax_path.yaxis.label.set_color('#D8CCFF')
    ax_path.title.set_color('#FFFFFF')
    ax_path.spines['bottom'].set_color('#4F359B')
    ax_path.spines['top'].set_color('#4F359B')
    ax_path.spines['left'].set_color('#4F359B')
    ax_path.spines['right'].set_color('#4F359B')
    
    ax_path.set_xlim(-0.5, factory_w - 0.5)
    ax_path.set_ylim(-0.5, factory_h - 0.5)
    ax_path.set_xticks(range(factory_w))
    ax_path.set_yticks(range(factory_h))
    ax_path.grid(True, linestyle='--', alpha=0.3, color='#FFA500') # Orange grid
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
                                          facecolor=face_color, alpha=0.8)
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

    # --- Visualization 2: Congestion Heatmap ---
    fig_heat, ax_heat = plt.subplots(1, figsize=(max(10, factory_w/2.5), max(10, factory_h/2.8))) 
    
    # Style for dark mode
    fig_heat.patch.set_facecolor('#4F359B') # Main BG
    ax_heat.set_facecolor('#3E2C5E') # Plot BG
    ax_heat.tick_params(colors='#D8CCFF')
    ax_heat.xaxis.label.set_color('#D8CCFF')
    ax_heat.yaxis.label.set_color('#D8CCFF')
    ax_heat.title.set_color('#FFFFFF')
    ax_heat.spines['bottom'].set_color('#4F359B')
    ax_heat.spines['top'].set_color('#4F359B')
    ax_heat.spines['left'].set_color('#4F359B')
    ax_heat.spines['right'].set_color('#4F359B')

    ax_heat.set_xlim(-0.5, factory_w - 0.5)
    ax_heat.set_ylim(-0.5, factory_h - 0.5)
    ax_heat.set_xticks(range(factory_w))
    ax_heat.set_yticks(range(factory_h))
    ax_heat.grid(True, linestyle='--', alpha=0.3, color='#FFA500') # Orange grid
    ax_heat.set_aspect('equal', adjustable='box')
    ax_heat.invert_yaxis() 

    max_flow = max(max(row) for row in flow_density_grid) if flow_density_grid and any(any(row) for row in flow_density_grid) else 1
    flow_density_grid_transposed = [list(row) for row in zip(*flow_density_grid)]

    # Draw Heatmap
    if max_flow > 0:
        colors = [(0, 0, 0, 0), (0.1, 0.7, 0.1, 0.5), (1, 0.5, 0, 0.8), (1, 0, 0, 1)] 
        cmap_custom = LinearSegmentedColormap.from_list("flow_cmap", colors)
        
        im = ax_heat.imshow(flow_density_grid_transposed, cmap=cmap_custom, 
                            interpolation='nearest', origin='lower', extent=[-0.5, factory_w-0.5, -0.5, factory_h-0.5],
                            vmax=max_flow)
        
        cbar = fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Material Flow Density (Path Overlaps)')
        plt.setp(plt.getp(cbar.ax.yaxis, 'ticklabels'), color='#D8CCFF')
        cbar.ax.yaxis.label.set_color('#D8CCFF')


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
                                          facecolor=face_color, alpha=0.8)
            ax_heat.add_patch(rect_body)
            ax_heat.text(x + width/2 - 0.5, y + height/2 - 0.5, f"M{machine_id}",
                         ha='center', va='center', fontsize=7, color='white', weight='bold')


    ax_heat.set_title("Factory Congestion Heatmap (A* Flow Analysis)", fontsize=14)
    ax_heat.set_xlabel("Factory Width (X)")
    ax_heat.set_ylabel("Factory Height (Y)")
    ax_heat.invert_yaxis() 
    
    return fig_path, fig_heat

def generate_performance_summary(ga_metrics, a_star_metrics, process_sequence, machines_defs, target_tph):
    """This function is designed to be captured by redirect_stdout."""
    
    machine_util_data_str = ga_metrics.get("machine_utilization", {})
    machine_util_data = {int(k): v for k, v in machine_util_data_str.items()} 
    
    util_rows = []
    machines_dict = {m['id']: m for m in machines_defs}
    
    bottleneck_mid = None
    max_util = 0.0
    
    for mid in process_sequence:
        name = machines_dict.get(mid, {}).get("name", "N/A")
        util = machine_util_data.get(mid, 0.0)
        cycle_time = machines_dict.get(mid, {}).get("cycle_time", 0)
        
        util_rows.append((mid, name, cycle_time, util * 100))
        
        if util > max_util:
            max_util = util
            bottleneck_mid = mid

    print("\n" + "="*80)
    print(f"{'FACTORY LAYOUT OPTIMIZATION SUMMARY':^80}")
    print("="*80)
    print(f"{'Metric':<35} | {'GA Value':<20} | {'A* Flow Value':<20}")
    print("-" * 80)
    print(f"{'Best Fitness Score (GA)':<35} | {ga_metrics.get('fitness', 0.0):<20.2f} | {'-':<20}")
    print(f"{'Hourly Throughput (TPH)':<35} | {ga_metrics.get('throughput', 0.0):<20.2f} | {'-':<20}")
    print(f"{'Target TPH':<35} | {target_tph:<20.0f} | {'-':<20}")
    print(f"{'Total Distance (Euclidean)':<35} | {ga_metrics.get('distance', 0.0):<20.2f} | {'-':<20}")
    print(f"{'Total Flow Distance (A*)':<35} | {'-':<20} | {a_star_metrics.get('total_a_star_distance', 0.0):<20.1f}")
    print(f"{'Total Travel Time (A* Flow)':<35} | {'-':<20} | {a_star_metrics.get('total_a_star_time_seconds', 0.0):<20.2f}")
    print(f"{'Area Utilization Ratio':<35} | {ga_metrics.get('utilization_ratio', 0.0)*100:<20.2f}% | {'-':<20}")
    print(f"{'Zone 1 Proximity Penalty':<35} | {ga_metrics.get('zone_penalty', 0.0):<20.2f} | {'-':<20}")
    print("-" * 80)
    
    print(f"\n{'MACHINE UTILIZATION ANALYSIS':^80}")
    print("-" * 80)
    print(f"{'ID':<4} | {'Name':<30} | {'Cycle Time (s)':<15} | {'Utilization Rate':<20}")
    print("-" * 80)
    
    for mid, name, cycle_time, util_rate in util_rows:
        highlight = " 🌟 BOTTLENECK" if mid == bottleneck_mid else ""
        print(f"M{mid:<3} | {name:<30} | {cycle_time:<15} | {util_rate:<19.2f}%{highlight}")
    print("="*80 + "\n")

def run_pathfinding_analysis(layout_data, material_travel_speed):
    """Loads GA results from memory, calculates A* paths, and returns metrics/visuals."""
    
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            factory_w = layout_data["factory_width"]
            factory_h = layout_data["factory_height"]
            machine_positions_str = layout_data["machine_positions_map"]
            machine_positions = {int(k): v for k, v in machine_positions_str.items()}
            process_sequence = layout_data["process_sequence"]
            machines_definitions = layout_data["machines_definitions"]
            machines_dict = {m['id']: m for m in machines_definitions}
            ga_metrics = layout_data["final_metrics"]

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
                                obstacle_grid[i][j] = 1 

            print("\n🚀 Starting Single-Stroke Continuous Path Optimization...")
            all_paths_found = []
            access_points_cache = {} 
            
            flow_density_grid = [[0 for _ in range(factory_h)] for _ in range(factory_w)]
            total_a_star_distance = 0
            total_a_star_time = 0

            first_machine_id = process_sequence[0]
            first_pos_info = machine_positions.get(first_machine_id)
            first_machine_def = machines_dict.get(first_machine_id)
            if first_pos_info and first_machine_def:
                access_points_cache[first_machine_id] = get_best_access_point(
                    first_pos_info, first_machine_def, obstacle_grid, factory_w, factory_h)
            
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
                
                # *** THIS IS THE BUG FIX ***
                # Was: a_star_search(..., goal_coords, ...)
                # Now: a_star_search(..., goal_node, ...)
                path = a_star_search(obstacle_grid, start_node, goal_node, factory_w, factory_h)
                
                a_star_dist = 0
                travel_time = 0
                
                if path:
                    all_paths_found.append(path)
                    a_star_dist = len(path) - 1
                    travel_time = a_star_dist / material_travel_speed
                    
                    for x, y in path:
                        if 0 <= x < factory_w and 0 <= y < factory_h and obstacle_grid[x][y] == 0:
                            flow_density_grid[x][y] += 1
                else:
                    all_paths_found.append([start_node, goal_node])
                    a_star_dist = abs(start_node[0] - goal_node[0]) + abs(start_node[1] - goal_node[1]) 
                    travel_time = a_star_dist / material_travel_speed
                    
                total_a_star_distance += a_star_dist
                total_a_star_time += travel_time
                
                print(f"M{current_machine_id} -> M{next_machine_id:<7} | {str(start_node):<12} | {str(goal_node):<12} | {a_star_dist:<15.1f} | {travel_time:<15.2f}")

            print("-" * 70)
            print(f"{'A* Totals':<30} | {'':<12} | {'Total A* Distance':<15} | {total_a_star_distance:<15.1f}")
            print(f"{'':<30} | {'':<12} | {'Total Flow Time (s)':<15} | {total_a_star_time:<15.2f}")
            print("\n🎨 Visualizing Single-Stroke Optimization Result and Congestion Heatmap...")

            a_star_metrics = {
                "total_a_star_distance": total_a_star_distance,
                "total_a_star_time_seconds": total_a_star_time
            }

            # Generate visualizations
            fig_path, fig_heat = visualize_layout_with_paths(layout_data, all_paths_found, flow_density_grid)
            
            # Generate final summary text
            summary_text = io.StringIO()
            with redirect_stdout(summary_text):
                generate_performance_summary(ga_metrics, a_star_metrics, process_sequence, machines_definitions, layout_data["target_tph"])
            
            return {
                "a_star_metrics": a_star_metrics,
                "fig_path": fig_path,
                "fig_heat": fig_heat,
                "a_star_log": f.getvalue(),
                "summary_log": summary_text.getvalue()
            }

        except Exception as e:
            # Print the full traceback to the Streamlit log
            import traceback
            print("--- A* PATHFINDING ERROR TRACEBACK ---")
            traceback.print_exc()
            print("-----------------------------------------")
            st.error(f"Error during A* Pathfinding: {e}")
            return {
                "a_star_metrics": {}, "fig_path": None, "fig_heat": None, 
                "a_star_log": f.getvalue(), "summary_log": ""
            }

def generate_analysis_plots(logs, target_tph, target_util_thresh):
    """Generates the 6-panel GA performance plot. Returns a fig object."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#4F359B') # Dark background for the figure
    
    # Plot 1: Fitness Progress
    axs[0, 0].set_facecolor('#3E2C5E')
    axs[0, 0].plot(logs['best_fitness'], label='Best Fitness', color='#9F60FF')
    axs[0, 0].plot(logs['avg_fitness'], label='Average Fitness (Valid)', color='#00C49A', linestyle='--')
    axs[0, 0].set_xlabel('Generation'); axs[0, 0].set_ylabel('Fitness')
    axs[0, 0].set_title('GA Fitness Progress'); axs[0, 0].legend(fontsize=8); axs[0, 0].grid(True, linestyle=':', alpha=0.3, color='#FFA500')
    
    # Plot 2: Total Distance
    axs[0, 1].set_facecolor('#3E2C5E')
    plot_distances = [d if d != float('inf') else max(filter(lambda x: x!=float('inf'), logs['best_distance']), default=1000)*1.1 for d in logs['best_distance']]
    if not any(d != float('inf') for d in logs['best_distance']) and logs['best_distance']: plot_distances = [0] * len(logs['best_distance']) 
    axs[0, 1].plot(plot_distances, label='Best Individual Euclidean Distance', color='#38A1FF')
    axs[0, 1].set_xlabel('Generation'); axs[0, 1].set_ylabel('Total Distance')
    axs[0, 1].set_title('Best Individual Distance'); axs[0, 1].legend(fontsize=8); axs[0, 1].grid(True, linestyle=':', alpha=0.3, color='#FFA500')
    if any(d == float('inf') for d in logs['best_distance']): axs[0, 1].text(0.05, 0.95, "Note: 'inf' distances capped", transform=axs[0, 1].transAxes, fontsize=8, verticalalignment='top', color='#D8CCFF')

    # Plot 3: Throughput vs. Target
    axs[0, 2].set_facecolor('#3E2C5E')
    axs[0, 2].plot(logs['best_throughput'], label='Best Individual Throughput', color='#FF6347')
    axs[0, 2].axhline(y=target_tph, color='gray', linestyle=':', label=f'Target TPH ({target_tph})')
    axs[0, 2].set_xlabel('Generation'); axs[0, 2].set_ylabel('Throughput')
    axs[0, 2].set_title('Best Individual Throughput'); axs[0, 2].legend(fontsize=8); axs[0, 2].grid(True, linestyle=':', alpha=0.3, color='#FFA500')

    # Plot 4: Valid Individuals Ratio
    axs[1, 0].set_facecolor('#3E2C5E')
    axs[1, 0].plot(logs['valid_ratio'], label='Valid Individuals Ratio (%)', color='#DAA520')
    axs[1, 0].set_xlabel('Generation'); axs[1, 0].set_ylabel('Valid Ratio (%)'); axs[1, 0].set_ylim(0, 105)
    axs[1, 0].set_title('Valid Individuals Ratio'); axs[1, 0].legend(fontsize=8); axs[1, 0].grid(True, linestyle=':', alpha=0.3, color='#FFA500')

    # Plot 5: Zone Constraint Penalty
    axs[1, 1].set_facecolor('#3E2C5E')
    axs[1, 1].plot(logs['best_zone_penalty'], label='Zone Constraint Penalty (Best)', color='#FF8C00')
    axs[1, 1].set_xlabel('Generation'); axs[1, 1].set_ylabel('Penalty Value'); 
    axs[1, 1].set_title('Zone Proximity Penalty'); axs[1, 1].legend(fontsize=8); axs[1, 1].grid(True, linestyle=':', alpha=0.3, color='#FFA500')

    # Plot 6: Area Utilization Ratio
    axs[1, 2].set_facecolor('#3E2C5E')
    axs[1, 2].plot([r * 100 for r in logs['best_utilization']], label='Area Utilization (Best)', color='#ADFF2F')
    axs[1, 2].axhline(y=target_util_thresh * 100, color='gray', linestyle=':', label=f'Target Min ({target_util_thresh*100:.0f}%)')
    axs[1, 2].set_xlabel('Generation'); axs[1, 2].set_ylabel('Utilization (%)'); 
    axs[1, 2].set_title('Area Utilization Ratio'); axs[1, 2].legend(fontsize=8); axs[1, 2].grid(True, linestyle=':', alpha=0.3, color='#FFA500')
    
    # Apply dark mode styling to all axes
    for ax_row in axs:
        for ax in ax_row:
            ax.tick_params(colors='#D8CCFF')
            ax.xaxis.label.set_color('#D8CCFF')
            ax.yaxis.label.set_color('#D8CCFF')
            ax.title.set_color('#FFFFFF')
            ax.spines['bottom'].set_color('#4F359B')
            ax.spines['top'].set_color('#4F359B')
            ax.spines['left'].set_color('#4F359B')
            ax.spines['right'].set_color('#4F359B')
            legend = ax.get_legend()
            if legend:
                plt.setp(legend.get_texts(), color='#D8CCFF')

    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------------------------------------
# --------------------------------- MAIN STREAMLIT FUNCTION ------------------------------------------
# ----------------------------------------------------------------------------------------------------

def run_optimization_process(factory_w, factory_h, target_tph, material_travel_speed,
                             ga_params, fitness_weights, constraint_params,
                             machines_definitions, process_sequence,
                             progress_placeholder, status_text):
    
    status_text.text("Preparing for optimization...")
    
    machines_for_ga_processing_order = [next(m for m in machines_definitions if m["id"] == pid) for pid in process_sequence]
    
    status_text.text(f"Generating initial population (Size: {ga_params['population_size']})...")
    population = [create_individual(machines_for_ga_processing_order, factory_w, factory_h) for _ in range(ga_params['population_size'])]
    status_text.text("Initial population generation complete.")
    
    best_overall_fitness = -float('inf')
    best_overall_chromosome = None
    best_overall_metrics = {} 
    
    # Loggers for analysis plots
    logs = {
        'best_fitness': [], 'avg_fitness': [], 'best_distance': [],
        'best_throughput': [], 'valid_ratio': [], 'best_zone_penalty': [],
        'best_utilization': []
    }

    num_generations = ga_params['num_generations']
    for generation in range(1, num_generations + 1):

        all_eval_results = []
        current_gen_total_fitness = 0
        current_gen_valid_individuals_count = 0

        for chromo in population:
            eval_dict = calculate_fitness(
                chromo, machines_for_ga_processing_order, process_sequence,
                factory_w, factory_h, target_tph, material_travel_speed,
                machines_definitions, fitness_weights, constraint_params
            )
            all_eval_results.append((chromo, eval_dict))
            if eval_dict["is_valid"]:
                current_gen_total_fitness += eval_dict["fitness"]
                current_gen_valid_individuals_count += 1
        
        if not all_eval_results:
            st.error(f"Generation {generation}: No evaluation results! Algorithm halted.")
            break

        current_gen_best_item = max(all_eval_results, key=lambda item: item[1]['fitness'])
        current_gen_best_chromosome = current_gen_best_item[0]
        current_gen_best_eval_dict = current_gen_best_item[1]
        
        current_gen_best_fitness = current_gen_best_eval_dict['fitness']
        
        # Update logs
        logs['best_fitness'].append(current_gen_best_fitness)
        logs['best_distance'].append(current_gen_best_eval_dict['distance'])
        logs['best_throughput'].append(current_gen_best_eval_dict['throughput'])
        logs['best_zone_penalty'].append(current_gen_best_eval_dict.get('zone_penalty', 0.0))
        logs['best_utilization'].append(current_gen_best_eval_dict.get('utilization_ratio', 0.0))

        valid_ratio = (current_gen_valid_individuals_count / ga_params['population_size']) * 100 if ga_params['population_size'] > 0 else 0
        logs['valid_ratio'].append(valid_ratio)
        
        current_gen_avg_fitness = (current_gen_total_fitness / current_gen_valid_individuals_count) if current_gen_valid_individuals_count > 0 else -float('inf')
        logs['avg_fitness'].append(current_gen_avg_fitness)

        if current_gen_best_fitness > best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_chromosome = current_gen_best_chromosome
            best_overall_metrics = copy.deepcopy(current_gen_best_eval_dict) 
            status_text.text(
                f"🌟 Generation {generation}: New best fitness! {best_overall_fitness:.2f} "
                f"(Dist: {best_overall_metrics['distance']:.2f}, "
                f"TPH: {best_overall_metrics['throughput']:.2f}, "
                f"ZonePen: {best_overall_metrics['zone_penalty']:.2f})"
            )
        
        progress_placeholder.text(
            f"Generation {generation}/{num_generations} - "
            f"BestF: {current_gen_best_fitness:.2f}, "
            f"AvgF: {current_gen_avg_fitness:.2f}, "
            f"ValidRatio: {valid_ratio:.1f}%"
        )
        
        # --- Genetic Operators ---
        new_population = []
        all_eval_results.sort(key=lambda item: item[1]['fitness'], reverse=True)
        for i in range(ga_params['elitism_count']):
            if i < len(all_eval_results) and all_eval_results[i][1]["is_valid"]:
                new_population.append(all_eval_results[i][0])

        num_offspring_to_generate = ga_params['population_size'] - len(new_population)
        eligible_parents_for_selection = [item for item in all_eval_results if item[1]["is_valid"]]
        if not eligible_parents_for_selection: eligible_parents_for_selection = all_eval_results 

        current_offspring_count = 0
        while current_offspring_count < num_offspring_to_generate:
            if not eligible_parents_for_selection: break
            parent1_chromo = selection(eligible_parents_for_selection, ga_params['tournament_size'])
            parent2_chromo = selection(eligible_parents_for_selection, ga_params['tournament_size'])
            
            if parent1_chromo is None or parent2_chromo is None: 
                if eligible_parents_for_selection:
                    if parent1_chromo is None: parent1_chromo = random.choice(eligible_parents_for_selection)[0]
                    if parent2_chromo is None: parent2_chromo = random.choice(eligible_parents_for_selection)[0]
                else: 
                    break

            child1_chromo, child2_chromo = crossover(parent1_chromo, parent2_chromo, ga_params['crossover_rate'])
            if random.random() < ga_params['mutation_rate']: child1_chromo = mutate(child1_chromo, machines_for_ga_processing_order, factory_w, factory_h, mutation_rate_per_gene=0.05)
            if random.random() < ga_params['mutation_rate']: child2_chromo = mutate(child2_chromo, machines_for_ga_processing_order, factory_w, factory_h, mutation_rate_per_gene=0.05)
            new_population.append(child1_chromo); current_offspring_count +=1
            if current_offspring_count < num_offspring_to_generate: new_population.append(child2_chromo); current_offspring_count +=1
        
        while len(new_population) < ga_params['population_size']:
            new_population.append(create_individual(machines_for_ga_processing_order, factory_w, factory_h))
        population = new_population[:ga_params['population_size']]

    # --- Final GA Result Processing ---
    status_text.text("Genetic Algorithm finished. Preparing final results...")
    
    if not best_overall_chromosome or not best_overall_metrics.get("is_valid"):
        st.error("Did not find a valid optimal layout.")
        return None

    final_grid_layout = initialize_layout_grid(factory_w, factory_h)
    final_machine_positions_map = {}
    
    for i, machine_def_item in enumerate(machines_for_ga_processing_order):
        pos_x_final, pos_y_final = best_overall_chromosome[i]
        place_machine_on_grid(final_grid_layout, machine_def_item["id"], machine_def_item["footprint"], pos_x_final, pos_y_final)
        final_machine_positions_map[machine_def_item["id"]] = {
            "x": pos_x_final, "y": pos_y_final,
            "center_x": pos_x_final + machine_def_item["footprint"][0] / 2.0,
            "center_y": pos_y_final + machine_def_item["footprint"][1] / 2.0,
        }
    
    # Capture layout text
    layout_log_io = io.StringIO()
    with redirect_stdout(layout_log_io):
        print_layout(final_grid_layout, final_machine_positions_map, factory_w, factory_h, process_sequence, machines_definitions)
    layout_log = layout_log_io.getvalue()
    
    # Generate GA layout visualization
    fig_ga_layout = visualize_layout_plt(
        final_grid_layout, final_machine_positions_map,
        factory_w, factory_h, process_sequence,
        machines_definitions, constraint_params
    )
    
    # Generate GA performance graphs
    fig_ga_analysis = generate_analysis_plots(logs, target_tph, constraint_params['area_util_min_threshold'])

    # Prepare data for pathfinding (in-memory)
    layout_data = {
        "factory_width": factory_w,
        "factory_height": factory_h,
        "process_sequence": process_sequence,
        "machines_definitions": machines_definitions,
        "machine_positions_map": {str(k): v for k, v in final_machine_positions_map.items()},
        "final_metrics": best_overall_metrics,
        "target_tph": target_tph # Pass this for the summary
    }

    # --- Run Pathfinding Analysis ---
    status_text.text("Running A* Pathfinding and Congestion Analysis...")
    pathfinding_results = run_pathfinding_analysis(layout_data, material_travel_speed)
    status_text.text("Analysis Complete!")
    
    return {
        "ga_metrics": best_overall_metrics,
        "a_star_metrics": pathfinding_results["a_star_metrics"],
        "layout_log": layout_log,
        "a_star_log": pathfinding_results["a_star_log"],
        "summary_log": pathfinding_results["summary_log"],
        "fig_ga_layout": fig_ga_layout,
        "fig_ga_analysis": fig_ga_analysis,
        "fig_path": pathfinding_results["fig_path"],
        "fig_heat": pathfinding_results["fig_heat"]
    }

# ----------------------------------------------------------------------------------------------------
# ------------------------------------ STREAMLIT UI DEFINITION ---------------------------------------
# ----------------------------------------------------------------------------------------------------

# --- Sidebar Inputs ---
st.sidebar.title("Configuration")

# Factory Settings
st.sidebar.header("🏭 Factory Settings")
# *** THESE VALUES ARE REDUCED TO PREVENT CRASHING ON STREAMLIT CLOUD ***
factory_w = st.sidebar.number_input("Factory Width (units)", min_value=10, max_value=100, value=20) # Reduced from 28
factory_h = st.sidebar.number_input("Factory Height (units)", min_value=10, max_value=100, value=20) # Reduced from 28
target_tph = st.sidebar.number_input("Target Production (units/hr)", min_value=1, max_value=200, value=35)
material_travel_speed = st.sidebar.slider("Material Travel Speed (units/sec)", 0.1, 5.0, 0.5, 0.1)

# GA Hyperparameters
with st.sidebar.expander("🧬 GA Hyperparameters", expanded=False):
    # *** THESE VALUES ARE REDUCED TO PREVENT CRASHING ON STREAMLIT CLOUD ***
    population_size = st.sidebar.number_input("Population Size", min_value=10, max_value=1000, value=20)  # Reduced from 40
    num_generations = st.sidebar.number_input("Number of Generations", min_value=10, max_value=2000, value=10)  # Reduced from 25
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.5, 0.01)
    crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8, 0.01)
    elitism_count = st.sidebar.number_input("Elitism Count", min_value=1, max_value=20, value=2)
    tournament_size = st.sidebar.number_input("Tournament Size", min_value=2, max_value=20, value=3)

# Fitness Weights
with st.sidebar.expander("⚖️ Fitness Weights", expanded=False):
    throughput_weight = st.sidebar.number_input("Throughput Weight", value=1.0)
    distance_weight = st.sidebar.number_input("Distance Weight", value=0.005, format="%.4f")
    zone_penalty_weight = st.sidebar.number_input("Zone Penalty Weight", value=0.5)
    mhs_turn_penalty_weight = st.sidebar.number_input("MHS Turn Penalty Weight", value=0.001, format="%.4f")
    utilization_bonus_weight = st.sidebar.number_input("Area Utilization Bonus Weight", value=0.1)
    bonus_for_target = st.sidebar.number_input("Bonus Factor (Target TPH)", value=0.2)

# Constraint Parameters
with st.sidebar.expander("⛓️ Constraint Parameters", expanded=False):
    zone_1_targets_json = st.sidebar.text_input("Zone 1 Target Machine IDs (JSON)", value=DEFAULT_ZONE_1_TARGETS_JSON)
    zone_1_max_spread = st.sidebar.number_input("Zone 1 Max Spread Distance", min_value=1.0, max_value=100.0, value=12.0)
    area_util_min = st.sidebar.slider("Min Area Utilization Threshold", 0.0, 1.0, 0.40, 0.01)

# Machine Data
with st.sidebar.expander("🤖 Machine & Process Data", expanded=True):
    machines_json = st.sidebar.text_area("Machines Definitions", value=DEFAULT_MACHINES_JSON, height=300)
    process_seq_json = st.sidebar.text_area("Process Sequence", value=DEFAULT_PROCESS_SEQUENCE_JSON, height=50)

# Run Button
run_button = st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True)

# --- Main Page Output Area ---
st.header("📊 Optimization Results")

# Setup tabs
tab_summary, tab_ga_layout, tab_astar_path, tab_heatmap, tab_ga_graphs, tab_logs = st.tabs([
    "📈 Summary", 
    "🏭 GA Layout", 
    "➡️ A* Paths", 
    "🔥 Congestion Heatmap", 
    "📉 GA Graphs",
    "🖨️ Raw Logs"
])

# Placeholders for live updates
progress_placeholder = st.empty()
status_text = st.empty()


if run_button:
    # 1. Clear previous results
    with tab_summary:
        st.empty()
    with tab_ga_layout:
        st.empty()
    with tab_astar_path:
        st.empty()
    with tab_heatmap:
        st.empty()
    with tab_ga_graphs:
        st.empty()
    with tab_logs:
        st.empty()

    # 2. Parse inputs
    try:
        machines_definitions = json.loads(machines_json)
        process_sequence = json.loads(process_seq_json)
        zone_1_target_machines = json.loads(zone_1_targets_json)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON input: {e}")
        st.stop()
    
    # 3. Bundle parameters
    ga_params = {
        "population_size": population_size,
        "num_generations": num_generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "elitism_count": elitism_count,
        "tournament_size": tournament_size
    }
    fitness_weights = {
        "throughput": throughput_weight,
        "distance": distance_weight,
        "zone_penalty": zone_penalty_weight,
        "mhs_turn_penalty": mhs_turn_penalty_weight,
        "utilization_bonus": utilization_bonus_weight,
        "bonus_for_target_achievement": bonus_for_target
    }
    constraint_params = {
        "zone_1_target_machines": zone_1_target_machines,
        "zone_1_max_spread_distance": zone_1_max_spread,
        "area_util_min_threshold": area_util_min
    }

    # 4. Run the main process
    with st.spinner("🚀 Running Optimization... This may take a moment..."):
        results = run_optimization_process(
            factory_w, factory_h, target_tph, material_travel_speed,
            ga_params, fitness_weights, constraint_params,
            machines_definitions, process_sequence,
            progress_placeholder, status_text
        )
    
    progress_placeholder.empty()
    status_text.empty()

    # 5. Display results
    if results:
        st.success("🎉 Optimization Complete!")
        st.balloons()
        
        ga_metrics = results["ga_metrics"]
        a_star_metrics = results["a_star_metrics"]

        with tab_summary:
            st.header("Key Performance Indicators")
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Fitness Score", f"{ga_metrics.get('fitness', 0.0):.2f}")
            col2.metric("Hourly Throughput (TPH)", f"{ga_metrics.get('throughput', 0.0):.2f}", f"{ga_metrics.get('throughput', 0.0) - target_tph:.2f} vs. Target")
            col3.metric("A* Flow Distance", f"{a_star_metrics.get('total_a_star_distance', 0.0):.1f} units")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Euclidean Distance", f"{ga_metrics.get('distance', 0.0):.2f} units")
            col5.metric("Area Utilization", f"{ga_metrics.get('utilization_ratio', 0.0):.2%}")
            col6.metric("Zone 1 Penalty", f"{ga_metrics.get('zone_penalty', 0.0):.2f}")

            st.header("Performance Summary")
            st.code(results["summary_log"])
        
        with tab_ga_layout:
            st.header("Best Layout (from Genetic Algorithm)")
            st.pyplot(results["fig_ga_layout"])
        
        with tab_astar_path:
            st.header("A* Material Flow Paths")
            st.pyplot(results["fig_path"])
        
        with tab_heatmap:
            st.header("Congestion Heatmap (from A* Flow)")
            st.pyplot(results["fig_heat"])
        
        with tab_ga_graphs:
            st.header("Genetic Algorithm Performance Graphs")
            st.pyplot(results["fig_ga_analysis"])

        with tab_logs:
            st.header("Raw Layout Log")
            st.code(results["layout_log"])
            st.header("A* Pathfinding Log")
            st.code(results["a_star_log"])
    else:
        st.error("Optimization failed to produce a valid result.")
