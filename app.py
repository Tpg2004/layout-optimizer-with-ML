import streamlit as st
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
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time

# --- App Configuration ---
st.set_page_config(
    page_title="Factory Layout Optimizer 🏭",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Core Optimization & Visualization Code (Adapted from your script) ---
# Note: Most functions are kept identical, with minor tweaks for Streamlit integration.

# Utility Classes & Functions
class Machine:
    def __init__(self, id, name, footprint, cycle_time=0, clearance=0, wall_affinity=False):
        self.id = id; self.name = name; self.footprint = footprint; self.cycle_time = cycle_time
        self.clearance = clearance; self.wall_affinity = wall_affinity; self.position = None

# --- Core GA & Pathfinding Logic ---
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

def calculate_total_distance(machine_positions, process_sequence, distance_type='euclidean'):
    total_distance = 0
    if not machine_positions or len(process_sequence) < 2: return float('inf')
    for i in range(len(process_sequence) - 1):
        m1_id, m2_id = process_sequence[i], process_sequence[i+1]
        pos1, pos2 = machine_positions.get(m1_id), machine_positions.get(m2_id)
        if not pos1 or not pos2: return float('inf')
        dx, dy = pos1['center_x'] - pos2['center_x'], pos1['center_y'] - pos2['center_y']
        if distance_type == 'manhattan':
            total_distance += abs(dx) + abs(dy)
        else:
            total_distance += math.sqrt(dx**2 + dy**2)
    return total_distance

def calculate_area_metrics(machine_positions, machines_defs_ordered_by_proc_seq, factory_w, factory_h):
    total_footprint_area = 0
    for machine_def in machines_defs_ordered_by_proc_seq:
        if machine_def["id"] in machine_positions:
            w, h = machine_def["footprint"]
            total_footprint_area += w * h
    factory_area = factory_w * factory_h
    return total_footprint_area, total_footprint_area / factory_area

def calculate_zone_penalty(machine_positions, zone_target_ids, max_spread_dist):
    zone_machines_pos = [machine_positions[m_id] for m_id in zone_target_ids if m_id in machine_positions]
    if len(zone_machines_pos) < 2: return 0.0
    max_dist = 0.0
    for i in range(len(zone_machines_pos)):
        for j in range(i + 1, len(zone_machines_pos)):
            pos1, pos2 = zone_machines_pos[i], zone_machines_pos[j]
            dist = math.sqrt((pos1['center_x'] - pos2['center_x'])**2 + (pos1['center_y'] - pos2['center_y'])**2)
            if dist > max_dist: max_dist = dist
    if max_dist > max_spread_dist:
        return (max_dist - max_spread_dist) ** 2
    return 0.0

def calculate_machine_utilization_and_bottleneck(machine_positions, process_sequence, all_machines_data, travel_speed, seconds_per_hour):
    if not machine_positions or not process_sequence: return (0.0, {})
    stage_times = {}
    for i in range(len(process_sequence)):
        current_machine_id = process_sequence[i]
        machine_cycle_time = get_machine_cycle_time(current_machine_id, all_machines_data)
        travel_time_to_next = 0.0
        if i < len(process_sequence) - 1:
            next_machine_id = process_sequence[i+1]
            pos_curr, pos_next = machine_positions.get(current_machine_id), machine_positions.get(next_machine_id)
            if pos_curr and pos_next:
                distance = math.sqrt((pos_curr['center_x'] - pos_next['center_x'])**2 + (pos_curr['center_y'] - pos_next['center_y'])**2)
                if travel_speed > 0:
                    travel_time_to_next = distance / travel_speed
                else:
                    travel_time_to_next = float('inf')
        stage_times[current_machine_id] = machine_cycle_time + travel_time_to_next
    max_stage_time = max(stage_times.values()) if stage_times else 0.0
    if max_stage_time <= 0 or max_stage_time == float('inf'): return (0.0, {})
    utilization_data = {}
    for machine_id in process_sequence:
        machine_cycle_time = get_machine_cycle_time(machine_id, all_machines_data)
        if machine_cycle_time == float('inf') or max_stage_time == 0.0:
            utilization_data[machine_id] = 0.0
        else:
            utilization_data[machine_id] = min(1.0, machine_cycle_time / max_stage_time)
    return (max_stage_time, utilization_data)

def calculate_fitness(chromosome, machines_defs_ordered_by_proc_seq, process_seq_ids,
                      factory_w, factory_h, target_prod_throughput, material_travel_speed,
                      seconds_per_hour, weights, bonus_factor, zone_params, area_util_min_thresh):
    grid = initialize_layout_grid(factory_w, factory_h)
    machine_positions = {}
    result = {"fitness": -float('inf'), "distance": float('inf'), "throughput": 0.0, "is_valid": False,
              "zone_penalty": 0.0, "mhs_turn_penalty": 0.0, "utilization_ratio": 0.0, "machine_utilization": {}}
    if len(chromosome) != len(machines_defs_ordered_by_proc_seq): return result

    for i, machine_def in enumerate(machines_defs_ordered_by_proc_seq):
        pos_x, pos_y = chromosome[i]
        if pos_x == -1 and pos_y == -1: return result
        m_footprint, m_clearance, m_id = machine_def["footprint"], machine_def.get("clearance", 0), machine_def["id"]
        if not can_place_machine(grid, m_footprint, m_clearance, pos_x, pos_y, factory_w, factory_h):
            return result
        place_machine_on_grid(grid, m_id, m_footprint, pos_x, pos_y)
        machine_positions[m_id] = {"x": pos_x, "y": pos_y, "center_x": pos_x + m_footprint[0]/2.0, "center_y": pos_y + m_footprint[1]/2.0}

    max_stage_time, utilization_data = calculate_machine_utilization_and_bottleneck(
        machine_positions, process_seq_ids, machines_defs_ordered_by_proc_seq, material_travel_speed, seconds_per_hour)
    result["machine_utilization"] = utilization_data
    throughput = seconds_per_hour / max_stage_time if max_stage_time > 0 and max_stage_time != float('inf') else 0.0
    if max_stage_time == float('inf') or throughput == 0.0: return result

    total_euclidean_dist = calculate_total_distance(machine_positions, process_seq_ids, 'euclidean')
    zone_penalty = calculate_zone_penalty(machine_positions, zone_params['target_machines'], zone_params['max_spread_dist'])
    _, utilization_ratio = calculate_area_metrics(machine_positions, machines_defs_ordered_by_proc_seq, factory_w, factory_h)
    utilization_bonus = (utilization_ratio - area_util_min_thresh) * factory_w * factory_h * weights['utilization_bonus'] if utilization_ratio >= area_util_min_thresh else 0.0
    total_manhattan_dist = calculate_total_distance(machine_positions, process_seq_ids, 'manhattan')
    mhs_turn_penalty_value = total_manhattan_dist * weights['mhs_turn_penalty']

    fitness_val = (weights['throughput'] * throughput) \
                  - (weights['distance'] * total_euclidean_dist) \
                  - (weights['zone_penalty'] * zone_penalty) \
                  - mhs_turn_penalty_value \
                  + utilization_bonus
    if throughput >= target_prod_throughput:
        fitness_val += throughput * bonus_factor

    result.update({"fitness": fitness_val, "distance": total_euclidean_dist, "throughput": throughput, "is_valid": True,
                   "zone_penalty": zone_penalty, "mhs_turn_penalty": mhs_turn_penalty_value, "utilization_ratio": utilization_ratio})
    return result

def selection(population_with_eval_results, tournament_size):
    actual_tournament_size = min(tournament_size, len(population_with_eval_results))
    if not population_with_eval_results or actual_tournament_size == 0: return None
    tournament = random.sample(population_with_eval_results, actual_tournament_size)
    return max(tournament, key=lambda item: item[1]['fitness'])[0]

def crossover(parent1_chromo, parent2_chromo, crossover_rate):
    child1, child2 = list(parent1_chromo), list(parent2_chromo)
    if random.random() < crossover_rate and len(parent1_chromo) > 1:
        cut_point = random.randint(1, len(parent1_chromo) - 1)
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

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_machine_access_points(machine_pos_info, machine_def, obstacle_grid, factory_w, factory_h):
    footprint_w, footprint_h = machine_def["footprint"]
    start_x, start_y = machine_pos_info["x"], machine_pos_info["y"]
    access_points = []
    for x in range(start_x, start_x + footprint_w):
        for dy in [-1, footprint_h]:
            y = start_y + dy
            if 0 <= x < factory_w and 0 <= y < factory_h and obstacle_grid[x][y] == 0:
                access_points.append((x, y))
    for y in range(start_y, start_y + footprint_h):
        for dx in [-1, footprint_w]:
            x = start_x + dx
            if 0 <= x < factory_w and 0 <= y < factory_h and obstacle_grid[x][y] == 0:
                access_points.append((x, y))
    return list(set(access_points))

def get_best_access_point(machine_pos_info, machine_def, obstacle_grid, factory_w, factory_h):
    access_points = find_machine_access_points(machine_pos_info, machine_def, obstacle_grid, factory_w, factory_h)
    if not access_points:
        center_x, center_y = int(round(machine_pos_info["center_x"])), int(round(machine_pos_info["center_y"]))
        return (center_x, center_y) # Fallback
    center_point = (machine_pos_info["center_x"], machine_pos_info["center_y"])
    return min(access_points, key=lambda p: abs(p[0] - center_point[0]) + abs(p[1] - center_point[1]))

def get_optimized_access_point_for_sequence(prev_access_point, current_pos_info, current_machine_def,
                                            next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h):
    current_access_points = find_machine_access_points(current_pos_info, current_machine_def, obstacle_grid, factory_w, factory_h)
    if not current_access_points:
        return get_best_access_point(current_pos_info, current_machine_def, obstacle_grid, factory_w, factory_h)
    next_target = get_best_access_point(next_pos_info, next_machine_def, obstacle_grid, factory_w, factory_h)
    best_point = min(current_access_points, key=lambda p: (abs(prev_access_point[0] - p[0]) + abs(prev_access_point[1] - p[1])) + (abs(p[0] - next_target[0]) + abs(p[1] - next_target[1])))
    return best_point

def a_star_search(grid_map, start_coords, goal_coords, factory_w, factory_h):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start_coords: 0}
    fscore = {start_coords: heuristic(start_coords, goal_coords)}
    oheap = [(fscore[start_coords], start_coords)]
    while oheap:
        _, current_node = heapq.heappop(oheap)
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
            if not (0 <= neighbor[0] < factory_w and 0 <= neighbor[1] < factory_h) or grid_map[neighbor[0]][neighbor[1]] == 1:
                continue
            tentative_g_score = gscore[current_node] + 1
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current_node
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal_coords)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None

# --- Visualization Functions ---
def visualize_layout_plt(machine_positions_map, factory_w, factory_h, process_sequence_list, machine_definitions_list, zone_params):
    fig, ax = plt.subplots(figsize=(factory_w/2.5, factory_h/2.5))
    ax.set_xlim(-0.5, factory_w - 0.5); ax.set_ylim(-0.5, factory_h - 0.5)
    ax.set_xticks(range(factory_w)); ax.set_yticks(range(factory_h))
    ax.grid(True, linestyle='--', alpha=0.7); ax.set_aspect('equal', adjustable='box')

    cmap = plt.colormaps.get_cmap('viridis')
    machines_dict_by_id = {m['id']: m for m in machine_definitions_list}

    # Draw Zone Constraint Boundary
    zone_centers = [machine_positions_map[m_id] for m_id in zone_params['target_machines'] if m_id in machine_positions_map]
    if len(zone_centers) > 1:
        center_x_coords = [c['center_x'] for c in zone_centers]
        center_y_coords = [c['center_y'] for c in zone_centers]
        avg_cx, avg_cy = sum(center_x_coords) / len(zone_centers), sum(center_y_coords) / len(zone_centers)
        required_radius = zone_params['max_spread_dist'] / 2.0
        circle_required = patches.Circle((avg_cx - 0.5, avg_cy - 0.5), radius=required_radius,
                                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
                                         alpha=0.6, label=f"Zone Max Spread ({zone_params['max_spread_dist']})")
        ax.add_patch(circle_required)

    # Draw Machines
    for machine_id in process_sequence_list:
        if machine_id in machine_positions_map:
            pos_data = machine_positions_map[machine_id]
            machine_info = machines_dict_by_id.get(machine_id)
            if machine_info:
                x, y, (w, h), clr = pos_data['x'], pos_data['y'], machine_info['footprint'], machine_info.get('clearance', 0)
                color_val = machine_id / max(len(machine_definitions_list) - 1, 1)
                rect_body = patches.Rectangle((x - 0.5, y - 0.5), w, h, linewidth=1.5, edgecolor='black', facecolor=cmap(color_val), alpha=0.8)
                ax.add_patch(rect_body)
                ax.text(x + w/2 - 0.5, y + h/2 - 0.5, f"M{machine_id}", ha='center', va='center', fontsize=8, color='white', weight='bold')
                if clr > 0:
                    rect_clearance = patches.Rectangle((x-clr-0.5, y-clr-0.5), w+2*clr, h+2*clr, linewidth=1, edgecolor=cmap(color_val), facecolor='none', linestyle=':', alpha=0.5)
                    ax.add_patch(rect_clearance)

    plt.title("Optimized Factory Layout (GA)", fontsize=14, weight='bold')
    plt.xlabel("Factory Width (X)"); plt.ylabel("Factory Height (Y)")
    ax.invert_yaxis()
    return fig

def visualize_layout_with_paths(layout_data, all_paths, flow_density_grid):
    factory_w, factory_h = layout_data["factory_width"], layout_data["factory_height"]
    machine_positions_map = {int(k): v for k, v in layout_data["machine_positions_map"].items()}
    machine_definitions_list = layout_data["machines_definitions"]
    machines_dict_by_id = {m['id']: m for m in machine_definitions_list}
    
    # --- Figure 1: Path Overlay ---
    fig_path, ax_path = plt.subplots(figsize=(factory_w/2.5, factory_h/2.5))
    ax_path.set_xlim(-0.5, factory_w - 0.5); ax_path.set_ylim(-0.5, factory_h - 0.5)
    ax_path.set_xticks(range(factory_w)); ax_path.set_yticks(range(factory_h))
    ax_path.grid(True, linestyle='--', alpha=0.6)
    ax_path.set_aspect('equal', adjustable='box')

    # Draw Machines
    for machine_id, pos_data in machine_positions_map.items():
        machine_info = machines_dict_by_id.get(machine_id)
        if machine_info:
            x, y, (w, h) = pos_data['x'], pos_data['y'], machine_info['footprint']
            color_val = machine_id / max(len(machine_definitions_list) - 1, 1)
            rect_body = patches.Rectangle((x - 0.5, y - 0.5), w, h, linewidth=1.5, edgecolor='black', facecolor=plt.cm.viridis(color_val), alpha=0.7)
            ax_path.add_patch(rect_body)
            ax_path.text(x + w/2 - 0.5, y + h/2 - 0.5, f"M{machine_id}", ha='center', va='center', fontsize=8, color='white', weight='bold')

    # Draw Paths
    cmap_paths = plt.colormaps.get_cmap('cool')
    for i, path_segment in enumerate(all_paths):
        if path_segment:
            path_color = cmap_paths(i / max(len(all_paths) - 1, 1))
            path_xs, path_ys = [p[0] for p in path_segment], [p[1] for p in path_segment]
            ax_path.plot(path_xs, path_ys, color=path_color, linewidth=2.5, alpha=0.9, marker='o', markersize=3, label=f'Path {i+1}')

    ax_path.set_title("Material Flow Paths (A*)", fontsize=14, weight='bold')
    ax_path.set_xlabel("Factory Width (X)"); ax_path.set_ylabel("Factory Height (Y)")
    ax_path.invert_yaxis()

    # --- Figure 2: Congestion Heatmap ---
    fig_heat, ax_heat = plt.subplots(figsize=(factory_w/2.5, factory_h/2.5))
    ax_heat.set_xlim(-0.5, factory_w - 0.5); ax_heat.set_ylim(-0.5, factory_h - 0.5)
    ax_heat.set_xticks(range(factory_w)); ax_heat.set_yticks(range(factory_h))
    ax_heat.grid(False)
    ax_heat.set_aspect('equal', adjustable='box')
    
    flow_density_grid_transposed = [list(row) for row in zip(*flow_density_grid)]
    max_flow = max(max(row) for row in flow_density_grid) if any(any(row) for row in flow_density_grid) else 1
    colors = [(0,0,0,0), (0.1, 0.7, 0.1, 0.5), (1, 0.5, 0, 0.8), (1, 0, 0, 1)]
    cmap_custom = LinearSegmentedColormap.from_list("flow_cmap", colors)
    im = ax_heat.imshow(flow_density_grid_transposed, cmap=cmap_custom, interpolation='nearest', origin='lower', extent=[-0.5, factory_w-0.5, -0.5, factory_h-0.5], vmax=max_flow)
    cbar = fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label('Material Flow Density (Path Overlaps)')

    for machine_id, pos_data in machine_positions_map.items():
        machine_info = machines_dict_by_id.get(machine_id)
        if machine_info:
            x, y, (w, h) = pos_data['x'], pos_data['y'], machine_info['footprint']
            rect_body = patches.Rectangle((x-0.5, y-0.5), w, h, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)
            ax_heat.add_patch(rect_body)
            ax_heat.text(x + w/2 - 0.5, y + h/2 - 0.5, f"M{machine_id}", ha='center', va='center', fontsize=8, color='white', weight='bold')

    ax_heat.set_title("Factory Congestion Heatmap", fontsize=14, weight='bold')
    ax_heat.set_xlabel("Factory Width (X)"); ax_heat.set_ylabel("Factory Height (Y)")
    ax_heat.invert_yaxis()

    return fig_path, fig_heat

def generate_performance_plots(logs, target_tph, area_util_min_thresh):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot 1: Fitness
    axes[0].plot(logs['best_fitness'], label='Best Fitness', color='blue')
    axes[0].plot(logs['avg_fitness'], label='Average Fitness', color='cyan', linestyle='--')
    axes[0].set_title('GA Fitness Progress'); axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Fitness')
    axes[0].legend(); axes[0].grid(True, alpha=0.5)

    # Plot 2: Distance
    plot_distances = [d if d != float('inf') else None for d in logs['best_distance']]
    axes[1].plot(plot_distances, label='Best Euclidean Distance', color='green')
    axes[1].set_title('Best Individual Distance'); axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Total Distance')
    axes[1].legend(); axes[1].grid(True, alpha=0.5)

    # Plot 3: Throughput
    axes[2].plot(logs['best_throughput'], label='Best Throughput (TPH)', color='red')
    axes[2].axhline(y=target_tph, color='gray', linestyle=':', label=f'Target TPH ({target_tph})')
    axes[2].set_title('Best Individual Throughput'); axes[2].set_xlabel('Generation'); axes[2].set_ylabel('Throughput')
    axes[2].legend(); axes[2].grid(True, alpha=0.5)

    # Plot 4: Valid Ratio
    axes[3].plot(logs['valid_ratio'], label='Valid Individuals (%)', color='purple')
    axes[3].set_title('Valid Individuals Ratio'); axes[3].set_xlabel('Generation'); axes[3].set_ylabel('Ratio (%)'); axes[3].set_ylim(0, 105)
    axes[3].legend(); axes[3].grid(True, alpha=0.5)
    
    # Plot 5: Zone Penalty
    axes[4].plot(logs['best_zone_penalty'], label='Zone Penalty (Best)', color='orange')
    axes[4].set_title('Zone Proximity Penalty'); axes[4].set_xlabel('Generation'); axes[4].set_ylabel('Penalty Value')
    axes[4].legend(); axes[4].grid(True, alpha=0.5)
    
    # Plot 6: Area Utilization
    axes[5].plot([r * 100 for r in logs['best_utilization']], label='Area Utilization (Best)', color='brown')
    axes[5].axhline(y=area_util_min_thresh * 100, color='gray', linestyle=':', label=f'Target Min ({area_util_min_thresh*100:.0f}%)')
    axes[5].set_title('Area Utilization Ratio'); axes[5].set_xlabel('Generation'); axes[5].set_ylabel('Utilization (%)')
    axes[5].legend(); axes[5].grid(True, alpha=0.5)

    plt.tight_layout()
    return fig

# --- Main Application Logic ---
def run_genetic_algorithm(params, machines_defs_ordered_by_proc_seq, process_sequence, status_placeholder):
    population = [create_individual(machines_defs_ordered_by_proc_seq, params['factory_w'], params['factory_h']) for _ in range(params['pop_size'])]
    
    best_overall_fitness = -float('inf')
    best_overall_chromosome = None
    best_overall_metrics = {}
    
    logs = {
        'best_fitness': [], 'avg_fitness': [], 'best_distance': [], 'best_throughput': [],
        'valid_ratio': [], 'best_zone_penalty': [], 'best_utilization': []
    }
    
    progress_bar = status_placeholder.progress(0)
    log_area = status_placeholder.expander("Show Generation Logs", expanded=False)

    for generation in range(1, params['num_generations'] + 1):
        all_eval_results = []
        current_gen_total_fitness = 0
        current_gen_valid_count = 0
        
        for chromo in population:
            eval_dict = calculate_fitness(
                chromo, machines_defs_ordered_by_proc_seq, process_sequence,
                params['factory_w'], params['factory_h'], params['target_tph'], params['travel_speed'],
                params['seconds_per_hour'], params['weights'], params['bonus_factor'],
                params['zone_params'], params['area_util_min_thresh']
            )
            all_eval_results.append((chromo, eval_dict))
            if eval_dict["is_valid"]:
                current_gen_total_fitness += eval_dict["fitness"]
                current_gen_valid_count += 1
        
        if not all_eval_results: break
        
        current_gen_best_item = max(all_eval_results, key=lambda item: item[1]['fitness'])
        current_gen_best_eval_dict = current_gen_best_item[1]
        
        logs['best_fitness'].append(current_gen_best_eval_dict['fitness'])
        logs['best_distance'].append(current_gen_best_eval_dict['distance'])
        logs['best_throughput'].append(current_gen_best_eval_dict['throughput'])
        logs['best_zone_penalty'].append(current_gen_best_eval_dict['zone_penalty'])
        logs['best_utilization'].append(current_gen_best_eval_dict['utilization_ratio'])
        
        valid_ratio = (current_gen_valid_count / params['pop_size']) * 100
        logs['valid_ratio'].append(valid_ratio)
        
        avg_fitness = current_gen_total_fitness / current_gen_valid_count if current_gen_valid_count > 0 else -float('inf')
        logs['avg_fitness'].append(avg_fitness)

        if current_gen_best_eval_dict['fitness'] > best_overall_fitness:
            best_overall_fitness = current_gen_best_eval_dict['fitness']
            best_overall_chromosome = current_gen_best_item[0]
            best_overall_metrics = copy.deepcopy(current_gen_best_eval_dict)
        
        status_placeholder.text(f"🚀 Generation {generation}/{params['num_generations']} | Best Fitness: {best_overall_fitness:.2f}")
        log_area.write(f"Gen {generation}: Best Fitness={current_gen_best_eval_dict['fitness']:.2f}, Avg Fitness={avg_fitness:.2f}, Valid={valid_ratio:.1f}%")
        progress_bar.progress(generation / params['num_generations'])

        # Create next generation
        new_population = []
        all_eval_results.sort(key=lambda item: item[1]['fitness'], reverse=True)
        for i in range(params['elitism_count']):
            if i < len(all_eval_results) and all_eval_results[i][1]["is_valid"]:
                new_population.append(all_eval_results[i][0])
        
        eligible_parents = [item for item in all_eval_results if item[1]["is_valid"]]
        if not eligible_parents: eligible_parents = all_eval_results

        while len(new_population) < params['pop_size']:
            if not eligible_parents: break
            parent1 = selection(eligible_parents, params['tournament_size'])
            parent2 = selection(eligible_parents, params['tournament_size'])
            if parent1 is None or parent2 is None:
                parent1 = random.choice(eligible_parents)[0]
                parent2 = random.choice(eligible_parents)[0]
            
            child1, child2 = crossover(parent1, parent2, params['crossover_rate'])
            if random.random() < params['mutation_rate']:
                child1 = mutate(child1, machines_defs_ordered_by_proc_seq, params['factory_w'], params['factory_h'], 0.05)
            if random.random() < params['mutation_rate']:
                child2 = mutate(child2, machines_defs_ordered_by_proc_seq, params['factory_w'], params['factory_h'], 0.05)
            
            new_population.append(child1)
            if len(new_population) < params['pop_size']:
                new_population.append(child2)

        population = new_population

    return best_overall_chromosome, best_overall_metrics, logs

def run_pathfinding_analysis(ga_results, params):
    factory_w, factory_h = params['factory_w'], params['factory_h']
    machine_positions = {int(k): v for k, v in ga_results['machine_positions_map'].items()}
    process_sequence = ga_results['process_sequence']
    machines_definitions = ga_results['machines_definitions']
    machines_dict = {m['id']: m for m in machines_definitions}
    
    obstacle_grid = [[0] * factory_h for _ in range(factory_w)]
    for m_id, pos in machine_positions.items():
        m_def = machines_dict.get(m_id)
        if m_def:
            for i in range(pos['x'], pos['x'] + m_def['footprint'][0]):
                for j in range(pos['y'], pos['y'] + m_def['footprint'][1]):
                    if 0 <= i < factory_w and 0 <= j < factory_h:
                        obstacle_grid[i][j] = 1

    all_paths_found = []
    access_points_cache = {}
    flow_density_grid = [[0] * factory_h for _ in range(factory_w)]
    total_a_star_distance = 0
    total_a_star_time = 0

    for i in range(len(process_sequence) - 1):
        curr_id, next_id = process_sequence[i], process_sequence[i+1]
        curr_pos, next_pos = machine_positions.get(curr_id), machine_positions.get(next_id)
        curr_def, next_def = machines_dict.get(curr_id), machines_dict.get(next_id)
        if not (curr_pos and next_pos and curr_def and next_def): continue

        start_node = access_points_cache.get(curr_id)
        if start_node is None:
            start_node = get_best_access_point(curr_pos, curr_def, obstacle_grid, factory_w, factory_h)
            access_points_cache[curr_id] = start_node
        
        # Determine goal node with look-ahead optimization
        if i < len(process_sequence) - 2:
            next_next_id = process_sequence[i+2]
            next_next_pos = machine_positions.get(next_next_id)
            next_next_def = machines_dict.get(next_next_id)
            if next_next_pos and next_next_def:
                goal_node = get_optimized_access_point_for_sequence(start_node, next_pos, next_def, next_next_pos, next_next_def, obstacle_grid, factory_w, factory_h)
            else:
                goal_node = get_best_access_point(next_pos, next_def, obstacle_grid, factory_w, factory_h)
        else: # Last segment
            goal_node = get_best_access_point(next_pos, next_def, obstacle_grid, factory_w, factory_h)
        access_points_cache[next_id] = goal_node

        path = a_star_search(obstacle_grid, start_node, goal_node, factory_w, factory_h)
        if path:
            all_paths_found.append(path)
            dist = len(path) - 1
            total_a_star_distance += dist
            total_a_star_time += dist / params['travel_speed']
            for x, y in path:
                if 0 <= x < factory_w and 0 <= y < factory_h:
                    flow_density_grid[x][y] += 1
        else:
            all_paths_found.append([start_node, goal_node]) # Fallback path
            dist = heuristic(start_node, goal_node)
            total_a_star_distance += dist
            total_a_star_time += dist / params['travel_speed']

    a_star_metrics = {
        "total_a_star_distance": total_a_star_distance,
        "total_a_star_time_seconds": total_a_star_time
    }
    return all_paths_found, flow_density_grid, a_star_metrics

# --- Streamlit UI ---

# Initialize session state
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

# Default Machine Data
default_machines_data = [
    {"id": 0, "name": "Raw Material Input", "footprint_w": 2, "footprint_h": 2, "cycle_time": 20, "clearance": 1},
    {"id": 1, "name": "1st Cutting", "footprint_w": 3, "footprint_h": 3, "cycle_time": 35, "clearance": 1},
    {"id": 2, "name": "Milling Process", "footprint_w": 4, "footprint_h": 2, "cycle_time": 45, "clearance": 1},
    {"id": 3, "name": "Drilling", "footprint_w": 2, "footprint_h": 2, "cycle_time": 25, "clearance": 1},
    {"id": 4, "name": "Heat Treatment A", "footprint_w": 3, "footprint_h": 4, "cycle_time": 70, "clearance": 2},
    {"id": 5, "name": "Precision Machining A", "footprint_w": 3, "footprint_h": 2, "cycle_time": 40, "clearance": 1},
    {"id": 6, "name": "Assembly A", "footprint_w": 2, "footprint_h": 3, "cycle_time": 55, "clearance": 2},
    {"id": 7, "name": "Final Inspection A", "footprint_w": 1, "footprint_h": 2, "cycle_time": 15, "clearance": 1},
    {"id": 8, "name": "2nd Cutting", "footprint_w": 3, "footprint_h": 2, "cycle_time": 30, "clearance": 1},
    {"id": 9, "name": "Surface Treatment", "footprint_w": 2, "footprint_h": 4, "cycle_time": 50, "clearance": 2},
    {"id": 10, "name": "Washing Process 1", "footprint_w": 2, "footprint_h": 2, "cycle_time": 20, "clearance": 1},
    {"id": 11, "name": "Heat Treatment B", "footprint_w": 4, "footprint_h": 4, "cycle_time": 75, "clearance": 2},
    {"id": 12, "name": "Precision Machining B", "footprint_w": 2, "footprint_h": 3, "cycle_time": 42, "clearance": 1},
    {"id": 13, "name": "Component Assembly", "footprint_w": 3, "footprint_h": 3, "cycle_time": 60, "clearance": 1},
    {"id": 14, "name": "Quality Inspection B", "footprint_w": 2, "footprint_h": 1, "cycle_time": 18, "clearance": 1},
    {"id": 15, "name": "Packaging Line A", "footprint_w": 4, "footprint_h": 3, "cycle_time": 30, "clearance": 2},
]

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://i.imgur.com/v25JV2E.png", use_column_width=True) # A generic factory icon
    st.title("⚙️ Configuration")
    st.markdown("---")

    with st.expander("🏭 **Factory & Production**", expanded=True):
        factory_w = st.slider("Factory Width", 10, 100, 28)
        factory_h = st.slider("Factory Height", 10, 100, 28)
        target_tph = st.number_input("Target Production per Hour", 1, 200, 35)
        travel_speed = st.number_input("Material Travel Speed (units/sec)", 0.1, 10.0, 0.5, 0.1)
        
    with st.expander("🧬 **Genetic Algorithm**", expanded=True):
        pop_size = st.slider("Population Size", 50, 1000, 300, 50)
        num_generations = st.slider("Number of Generations", 50, 1000, 300, 50)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.5, 0.05)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8, 0.05)
        elitism_count = st.slider("Elitism Count", 1, 20, 5)
        tournament_size = st.slider("Tournament Size", 2, 20, 5)

    with st.expander("⚖️ **Fitness Weights**", expanded=False):
        w_throughput = st.number_input("Throughput Weight", 0.0, 5.0, 1.0)
        w_distance = st.number_input("Distance Weight", 0.0, 1.0, 0.005)
        w_zone = st.number_input("Zone Penalty Weight", 0.0, 2.0, 0.5)
        w_mhs = st.number_input("MHS Turn Penalty Weight", 0.0, 1.0, 0.001)
        w_util = st.number_input("Area Utilization Bonus Weight", 0.0, 1.0, 0.1)
        bonus_factor = st.number_input("Target Achievement Bonus Factor", 0.0, 1.0, 0.2)

    with st.expander("⛓️ **Constraints**", expanded=False):
        area_util_min_thresh = st.slider("Min. Area Utilization Threshold", 0.0, 1.0, 0.40, 0.05)
        zone_1_max_spread = st.slider("Zone 1 Max Spread Distance", 5.0, 50.0, 12.0, 0.5)
        # We will get target machines from the multiselect in the main panel

# --- Main Panel ---
st.title("🏭 GA-Powered Factory Layout Optimizer")
st.markdown("""
This application uses a **Genetic Algorithm (GA)** to optimize the placement of machines in a factory. 
The goal is to find a layout that maximizes **throughput**, minimizes **material travel distance**, and adheres to specific **spatial constraints**. 
After optimization, it uses the **A\* algorithm** to find and visualize the optimal material flow paths and identify potential congestion areas.
""")

# --- Machine & Sequence Configuration ---
st.header("1. Define Machines & Process Sequence")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader("Machine Definitions")
    uploaded_file = st.file_uploader("Or upload a machine configuration JSON", type="json")
    if uploaded_file is not None:
        try:
            uploaded_data = json.load(uploaded_file)
            # Adapt to the required DataFrame format
            adapted_data = []
            for item in uploaded_data:
                w, h = item.get("footprint", (0,0))
                adapted_data.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "footprint_w": w,
                    "footprint_h": h,
                    "cycle_time": item.get("cycle_time"),
                    "clearance": item.get("clearance")
                })
            edited_df = st.data_editor(
                pd.DataFrame(adapted_data),
                num_rows="dynamic",
                key="machine_editor_uploaded"
            )
        except Exception as e:
            st.error(f"Error parsing JSON file: {e}")
            edited_df = st.data_editor(
                pd.DataFrame(default_machines_data),
                num_rows="dynamic",
                key="machine_editor_default_error"
            )
    else:
         edited_df = st.data_editor(
            pd.DataFrame(default_machines_data),
            num_rows="dynamic",
            key="machine_editor_default"
        )

# Convert dataframe back to the required list of dicts format for the algorithm
machines_definitions = []
for index, row in edited_df.iterrows():
    if pd.notna(row['id']) and row['name']:
        machines_definitions.append({
            "id": int(row['id']),
            "name": row['name'],
            "footprint": (int(row['footprint_w']), int(row['footprint_h'])),
            "cycle_time": int(row['cycle_time']),
            "clearance": int(row['clearance'])
        })

with col2:
    st.subheader("Process Sequence & Zone Constraints")
    available_machines = {m['name']: m['id'] for m in machines_definitions}
    
    # Let user define the process sequence
    selected_machine_names = st.multiselect(
        "Define Process Sequence (Order matters)",
        options=available_machines.keys(),
        default=[m['name'] for m in machines_definitions]
    )
    process_sequence = [available_machines[name] for name in selected_machine_names]

    # Let user define zone constraints
    zone_1_target_names = st.multiselect(
        "Select Machines for 'Zone 1' Proximity Constraint",
        options=available_machines.keys(),
        default=["1st Cutting", "Milling Process", "Drilling", "2nd Cutting"]
    )
    zone_1_target_machines = [available_machines[name] for name in zone_1_target_names]


st.markdown("---")
st.header("2. Run Optimization")

if st.button("🚀 Start Optimization & Analysis", type="primary", use_container_width=True):
    st.session_state.run_clicked = True
    st.session_state.optimization_results = None

    # Prepare parameters for the run
    params = {
        'factory_w': factory_w, 'factory_h': factory_h, 'target_tph': target_tph, 'travel_speed': travel_speed,
        'seconds_per_hour': 3600, 'pop_size': pop_size, 'num_generations': num_generations,
        'mutation_rate': mutation_rate, 'crossover_rate': crossover_rate, 'elitism_count': elitism_count,
        'tournament_size': tournament_size, 'bonus_factor': bonus_factor, 'area_util_min_thresh': area_util_min_thresh,
        'weights': {'throughput': w_throughput, 'distance': w_distance, 'zone_penalty': w_zone,
                    'mhs_turn_penalty': w_mhs, 'utilization_bonus': w_util},
        'zone_params': {'target_machines': zone_1_target_machines, 'max_spread_dist': zone_1_max_spread}
    }
    
    # Ensure machine definitions are ordered by the selected process sequence
    try:
        machines_defs_ordered_by_proc_seq = [next(m for m in machines_definitions if m["id"] == pid) for pid in process_sequence]
    except StopIteration:
        st.error("Error: One or more machines in the process sequence are not defined in the machine definitions table. Please check your inputs.")
        st.stop()

    with st.status("Running Optimization...", expanded=True) as status:
        # --- Run GA ---
        st.write("### Phase 1: Running Genetic Algorithm...")
        ga_status_placeholder = st.empty()
        best_chromosome, best_metrics, logs = run_genetic_algorithm(params, machines_defs_ordered_by_proc_seq, process_sequence, ga_status_placeholder)
        
        if not best_chromosome or not best_metrics.get("is_valid"):
            status.update(label="GA failed to find a valid solution.", state="error", expanded=False)
            st.error("The Genetic Algorithm could not find a valid layout. Try adjusting parameters (e.g., increase factory size, reduce machine footprints, or run for more generations).")
        else:
            final_machine_positions_map = {}
            for i, machine_def in enumerate(machines_defs_ordered_by_proc_seq):
                pos_x, pos_y = best_chromosome[i]
                final_machine_positions_map[machine_def["id"]] = {
                    "x": pos_x, "y": pos_y,
                    "center_x": pos_x + machine_def["footprint"][0] / 2.0,
                    "center_y": pos_y + machine_def["footprint"][1] / 2.0,
                }
            
            ga_results_for_pathfinding = {
                "factory_width": factory_w, "factory_height": factory_h,
                "process_sequence": process_sequence, "machines_definitions": machines_definitions,
                "machine_positions_map": {str(k): v for k, v in final_machine_positions_map.items()},
                "final_metrics": best_metrics
            }

            # --- Run Pathfinding ---
            st.write("### Phase 2: Running A* Pathfinding Analysis...")
            time.sleep(1) # For better UX
            all_paths, flow_grid, a_star_metrics = run_pathfinding_analysis(ga_results_for_pathfinding, params)
            best_metrics.update(a_star_metrics)
            
            # --- Generate Visuals ---
            st.write("### Phase 3: Generating Visualizations...")
            time.sleep(1)
            fig_layout = visualize_layout_plt(final_machine_positions_map, factory_w, factory_h, process_sequence, machines_definitions, params['zone_params'])
            fig_path, fig_heat = visualize_layout_with_paths(ga_results_for_pathfinding, all_paths, flow_grid)
            fig_performance = generate_performance_plots(logs, target_tph, area_util_min_thresh)

            st.session_state.optimization_results = {
                "params": params,
                "metrics": best_metrics,
                "machine_definitions": machines_definitions,
                "final_layout_data": ga_results_for_pathfinding,
                "figures": {
                    "layout": fig_layout,
                    "path": fig_path,
                    "heatmap": fig_heat,
                    "performance": fig_performance
                }
            }
            status.update(label="Optimization Complete!", state="complete", expanded=False)

if st.session_state.run_clicked and st.session_state.optimization_results:
    results = st.session_state.optimization_results
    metrics = results['metrics']
    params = results['params']
    
    st.markdown("---")
    st.header("📈 Results & Analysis")
    st.balloons()
    
    # Create tabs for different output sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary & Metrics", "🖼️ Layout & Flow Visuals", "📉 Performance Graphs", "💾 Raw Data"])

    with tab1:
        st.subheader("Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        col1.metric("GA Fitness Score", f"{metrics.get('fitness', 0.0):.2f}")
        col2.metric("Hourly Throughput (TPH)", f"{metrics.get('throughput', 0.0):.2f}", f"Target: {params['target_tph']}")
        col3.metric("Area Utilization", f"{metrics.get('utilization_ratio', 0.0)*100:.2f}%", f"Min Target: {params['area_util_min_thresh']*100:.0f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distance & Flow Metrics")
            st.markdown(f"""
            - **Euclidean Distance (GA):** `{metrics.get('distance', 0.0):.2f}` units
            - **A* Path Distance (Flow):** `{metrics.get('total_a_star_distance', 0.0):.2f}` units
            - **Total A* Travel Time:** `{metrics.get('total_a_star_time_seconds', 0.0):.2f}` seconds
            """)

            st.subheader("Constraint Penalties")
            st.markdown(f"""
            - **Zone 1 Proximity Penalty:** `{metrics.get('zone_penalty', 0.0):.2f}`
            - **MHS Turn Penalty Component:** `{metrics.get('mhs_turn_penalty', 0.0):.3f}`
            """)
        
        with col2:
            st.subheader("Machine Utilization & Bottleneck Analysis")
            machine_util_data = {int(k): v for k, v in metrics.get("machine_utilization", {}).items()}
            machines_dict = {m['id']: m for m in results['machine_definitions']}
            
            util_rows = []
            bottleneck_mid = max(machine_util_data, key=machine_util_data.get) if machine_util_data else None

            for mid in process_sequence:
                name = machines_dict.get(mid, {}).get("name", "N/A")
                util = machine_util_data.get(mid, 0.0) * 100
                is_bottleneck = "  bottleneck" if mid == bottleneck_mid else ""
                util_rows.append({"Machine ID": f"M{mid}", "Name": name, "Utilization": util, "Note": is_bottleneck})
            
            util_df = pd.DataFrame(util_rows)
            st.dataframe(util_df.style.format({"Utilization": "{:.2f}%"})
                                      .highlight_max(subset="Utilization", color='lightcoral'),
                                      use_container_width=True)

    with tab2:
        st.subheader("Optimized Layout Visualization")
        st.pyplot(results['figures']['layout'])
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Material Flow Paths (A*)")
            st.pyplot(results['figures']['path'])
        with col2:
            st.subheader("Congestion Heatmap")
            st.pyplot(results['figures']['heatmap'])

    with tab3:
        st.subheader("Genetic Algorithm Performance Over Generations")
        st.pyplot(results['figures']['performance'])

    with tab4:
        st.subheader("Download Full Results")
        st.markdown("Click the button below to download a JSON file containing all the final layout data, machine positions, and performance metrics for external use.")
        
        # Prepare data for download
        json_string = json.dumps(results['final_layout_data'], indent=4)
        
        st.download_button(
            label="📥 Download Results as JSON",
            data=json_string,
            file_name="optimized_factory_layout.json",
            mime="application/json",
            use_container_width=True
        )
        
        with st.expander("Show JSON Data"):
            st.json(results['final_layout_data'])
