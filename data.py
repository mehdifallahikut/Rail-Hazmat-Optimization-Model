import random
from collections import deque
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

BLOCKS_FILE_PATH = 'blocks.txt'

Block = str
Car = str
BlockDict = Dict[Block, List[Car]]

LAMBDA_MATRIX = {
    'A': {'A': 0, 'B': 1, 'C': 1, 'D': 0, 'E': 1, 'F': 0},
    'B': {'A': 1, 'B': 0, 'C': 1, 'D': 0, 'E': 1, 'F': 0},
    'C': {'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': 1, 'F': 0},
    'D': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 1, 'F': 0},
    'E': {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 0, 'F': 0},
    'F': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
}
Dist_to_Engine = 5


def generate_blocks(num_blocks: int, hazmat_percentage: float, diversity_factor: float, train_length: int) -> Dict[
    str, List[str]]:
    if train_length < num_blocks:
        raise ValueError("Error")
    hazmat_types = ['A', 'B', 'C', 'D', 'E']
    blocks = {}
    used_names = set()
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    remaining_cars = train_length - num_blocks
    block_sizes = [1] * num_blocks
    for _ in range(remaining_cars):
        block_sizes[random.randint(0, num_blocks - 1)] += 1
    for i in range(num_blocks):
        while True:
            block_name = letters[i % len(letters)] * 2
            if block_name not in used_names:
                used_names.add(block_name)
                break
        total_cars = block_sizes[i]
        num_hazmat = int(total_cars * hazmat_percentage)
        num_unique = max(1, round(len(hazmat_types) * diversity_factor))
        selected = random.sample(hazmat_types, num_unique)
        hazmat_cars = [random.choice(selected) for _ in range(num_hazmat)]
        non_hazmat = ['F'] * (total_cars - num_hazmat)
        all_cars = hazmat_cars + non_hazmat
        random.shuffle(all_cars)
        blocks[block_name] = all_cars
    return blocks


def generate_rail_network(total_nodes: int, destinations: List[str], variance_level: float, train_length: int,
                          min_degree_extra: int = 1, min_degree_key: int = 5) -> nx.Graph:
    key_nodes = ['SS'] + destinations
    num_key_nodes = len(key_nodes)
    key_nodes_set = set(key_nodes)

    if total_nodes < num_key_nodes:
        raise ValueError("Node count error")

    num_extra_nodes = total_nodes - num_key_nodes
    extra_node_labels = [f'N{i + 1}' for i in range(num_extra_nodes)]

    # 1. Node Positioning
    pos = {}
    x_spacing = 10.0
    key_x_coords = [i * x_spacing for i in range(num_key_nodes)]
    for i, node in enumerate(key_nodes):
        pos[node] = (key_x_coords[i], random.uniform(-1.0, 1.0))

    if num_extra_nodes > 0:
        min_x = key_x_coords[0]
        max_x = key_x_coords[-1]
        y_range = max(3, num_key_nodes / 1.5)
        for node in extra_node_labels:
            while True:
                new_pos = (random.uniform(min_x, max_x), random.uniform(-y_range, y_range))
                if not any(np.linalg.norm(np.array(new_pos) - np.array(p)) < 2.0 for p in pos.values()):
                    pos[node] = new_pos
                    break

    # 2. Graph Construction
    G = nx.Graph()
    for node, p in pos.items(): G.add_node(node, pos=p)
    points = np.array(list(pos.values()))
    nodes_list = list(pos.keys())
    if len(points) >= 3:
        tri = Delaunay(points)
        for simplex in tri.simplices:
            for i in range(3):
                u, v = nodes_list[simplex[i]], nodes_list[simplex[(i + 1) % 3]]
                if u in key_nodes_set and v in key_nodes_set: continue
                G.add_edge(u, v)

    # 3. Connectivity
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        main_c = max(comps, key=len)
        for c in comps:
            if c != main_c:
                u = random.choice(list(c))
                v = min(main_c, key=lambda n: np.linalg.norm(np.array(pos[n]) - np.array(pos[u])))
                G.add_edge(u, v)

    # 4. Remove Direct Key Links & Add Mediators
    for u, v in [(u, v) for u, v in G.edges() if u in key_nodes_set and v in key_nodes_set]:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            if not nx.has_path(G, u, v):
                best_med = min(extra_node_labels,
                               key=lambda n: np.linalg.norm(np.array(pos[n]) - np.array(pos[u])) + np.linalg.norm(
                                   np.array(pos[n]) - np.array(pos[v])))
                G.add_edge(u, best_med);
                G.add_edge(v, best_med)

    # ============================================
    # 🎯 PARAMETERS & LOGIC
    # ============================================
    CITY_COST = 58.0
    CITY_EXPOSURE = 290.0
    TRAP_COST = 38.0
    TRAP_EXPOSURE = 200.0

    base_P = [random.uniform(0.018, 0.040) for _ in range(train_length)]
    length_penalty_factor = 1.15 if train_length > 20 else 1.0

    for u, v in G.edges():
        u_pos = np.array(pos[u])
        v_pos = np.array(pos[v])
        edge_length = np.linalg.norm(v_pos - u_pos)

        is_near_key = (u in key_nodes_set or v in key_nodes_set)
        is_long_edge = (edge_length > 5) and (edge_length < 15)

        # --- 1. ZIGZAG FIX: Mild Vertical Penalty ---
        # محاسبه میزان عمودی بودن یال
        dx = abs(u_pos[0] - v_pos[0])
        dy = abs(u_pos[1] - v_pos[1])
        penalty = 0.0


        if dy > dx * 2.5:
            penalty = 6.0

            # --- 2. STATION COLOR VARIETY ---
        zone_type = "NEUTRAL"

        if is_long_edge and not is_near_key and random.random() < 0.25:
            zone_type = "TRAP"
        elif is_near_key:
            if random.random() < 0.15:
                zone_type = "NEUTRAL"
            else:
                zone_type = "SAFE"
        else:
            zone_type = "NEUTRAL"

        # --- اعمال مقادیر ---
        if zone_type == "TRAP":
            final_cost = TRAP_COST + random.uniform(-2, 2) + penalty
            final_exposure = TRAP_EXPOSURE + random.uniform(-10, 10)
            P_values = []
            for i in range(train_length):
                pr = i / train_length
                m = 3.8 if pr < 0.3 else (1.8 if pr < 0.7 else 0.25)
                P_values.append(min(0.06, base_P[i] * m * length_penalty_factor))

        elif zone_type == "SAFE":  # City
            final_cost = CITY_COST + random.uniform(-2, 2) + penalty
            final_exposure = CITY_EXPOSURE + random.uniform(-15, 15)
            P_values = [max(0.001, base_P[i] * 0.05) for i in range(train_length)]

        else:  # NEUTRAL
            final_cost = 52.0 + random.uniform(-4, 4) + penalty
            final_exposure = 170.0 + random.uniform(-15, 15)
            P_values = [max(0.010, min(0.050, (base_P[i] + random.uniform(-0.005, 0.005)) * length_penalty_factor)) for
                        i in range(train_length)]

        G.edges[u, v]['P'] = P_values
        G.edges[u, v]['cost'] = round(final_cost, 1)
        G.edges[u, v]['exposure'] = int(round(final_exposure))

    return G