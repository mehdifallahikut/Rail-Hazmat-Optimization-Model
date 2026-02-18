# solution_utils.py
import networkx as nx
from typing import List, Dict, Tuple, Any
import networkx as nx
from data import LAMBDA_MATRIX,Dist_to_Engine

def precompute_min_risk_and_max_totals(G: nx.DiGraph, key_nodes: list[str], train_length: int):
    """
    min_risk_dict[pos][source][target] = min over all paths s->t of sum_{edges} exposure * P[pos-1]
    max_totals[p] = sum over all edges of exposure * P[p]
    """
    # --- defensively pad P to train_length ---
    for u, v, d in G.edges(data=True):
        P = d.get('P', [])
        if len(P) < train_length:
            # pad with zeros
            P = list(P) + [0.0] * (train_length - len(P))
            d['P'] = P

    # --- build min_risk_dict ---
    min_risk_dict: dict[int, dict[str, dict[str, float]]] = {}
    for pos in range(1, train_length + 1):
        def weight_func(u, v, d):
            P = d['P']
            return float(d.get('exposure', 0.0)) * float(P[pos - 1])
        min_risk_dict[pos] = {}
        for source in key_nodes:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight=weight_func)
            #    key_nodes و ≠source
            min_risk_dict[pos][source] = {
                t: lengths.get(t, float('inf')) for t in key_nodes if t != source
            }

    # --- compute max_totals safely ---
    max_totals = [0.0] * train_length
    for u, v, d in G.edges(data=True):
        exp = float(d.get('exposure', 0.0))
        P = d.get('P', [0.0] * train_length)
        for p in range(train_length):
            max_totals[p] += exp * P[p]

    return min_risk_dict, max_totals


def precompute_global_max(G, blocks, train_length, num_sections=None):
    if num_sections is None:
        num_sections = len(blocks)
    # Tighter max_cost: max over all possible paths (use all_pairs equivalent via nested max)
    max_cost_path = max(
        max(nx.single_source_dijkstra_path_length(G, source, weight='cost').values())
        for source in G.nodes
    )
    global_max_cost = num_sections * max_cost_path
    # Tighter max_risk: max per edge risk = exp * sum_P (assuming worst y=1 for all pos)
    max_per_edge_risk = max(
        d['exposure'] * sum(d['P'][:train_length]) for _, _, d in G.edges(data=True)
    )
    # Max path len tighter: average path len instead of max
    avg_path_len = (
        sum(len(nx.shortest_path(G, s, t)) for s in G.nodes for t in G.nodes if s != t)
        / (len(G.nodes) ** 2 - len(G.nodes))
    ) or 1
    num_hazmat = sum(1 for b in blocks.values() for car in b if car != 'F')
    global_max_risk = num_sections * avg_path_len * max_per_edge_risk * (num_hazmat / train_length)  # Normalize by density
    return global_max_cost, global_max_risk


def validate_cost_and_risk(perm, final_order, sections, lengths, paths, train_length, G
                           , debug=False):
    """
    Recomputes cost and risk independently using model outputs, and provides detailed risk per edge per position.

    Args:
    - perm: Tuple of block order.
    - final_order: Dict of final car order per block.
    - sections: List of (start, end) per section.
    - lengths: List of train lengths per section.
    - paths: Dict of section -> ordered node list.
    - blocks: Dict of blocks.
    - train_length: Integer.
    - G: nx.DiGraph.
    - global_max_cost: Float.
    - global_max_risk: Float.
    - debug: Bool for printing details.

    Returns:
    - dict: {'validated_cost': float, 'validated_risk': float, 'detailed_risk': dict}
    """
    if len(sections) != len(paths) or len(lengths) != len(sections):
        return {'validated_cost': None, 'validated_risk': None, 'detailed_risk': None}

    # Build full train types (car types in positions)
    train_types = []
    for b in perm:
        train_types += final_order.get(b, [])
    if len(train_types) != train_length:
        return {'validated_cost': None, 'validated_risk': None, 'detailed_risk': None}

    validated_cost = 0.0
    validated_risk = 0.0
    detailed_risk = {}  # section -> edge -> pos -> contrib

    for k in range(len(sections)):
        L = lengths[k]
        current_train_types = train_types[:L]  # Truncate to current length
        y = [1 if current_train_types[p - 1] != 'F' else 0 for p in range(1, L + 1)]

        section_path = paths.get(k, [])
        if not section_path:
            return {'validated_cost': None, 'validated_risk': None, 'detailed_risk': None}

        detailed_risk[k] = {}
        section_cost = 0.0
        section_risk = 0.0

        for i in range(len(section_path) - 1):
            u = section_path[i]
            v = section_path[i + 1]
            edge = (u, v)
            if not G.has_edge(u, v):
                return {'validated_cost': None, 'validated_risk': None, 'detailed_risk': None}

            data = G[u][v]
            cost = data.get('cost', 0.0)
            section_cost += cost

            exp = data.get('exposure', 0.0)
            P = data.get('P', [0.0] * train_length)

            detailed_risk[k][edge] = {}
            link_risk = 0.0
            for pos in range(1, L + 1):
                if y[pos - 1] == 1:
                    contrib = exp * P[pos - 1]
                    detailed_risk[k][edge][pos] = contrib
                    link_risk += contrib

            section_risk += link_risk

        validated_cost += section_cost
        validated_risk += section_risk

        if debug:
            print(
                f"Section {k} ({sections[k][0]} to {sections[k][1]}): Cost {section_cost:.4f}, Risk {section_risk:.4f}")
            for edge, pos_contrib in detailed_risk[k].items():
                print(f"  Edge {edge}:")
                for pos, contrib in pos_contrib.items():
                    print(f"    Pos {pos}: {contrib:.6f}")

    return {'validated_cost': validated_cost, 'validated_risk': validated_risk, 'detailed_risk': detailed_risk}


from tabulate import tabulate

def print_path_cost_risk_details(validated_results, sections, paths, G):
    if validated_results['validated_cost'] is None:
        print("No valid results to display.")
        return

    # Concatenate all edges, costs, and risks across all sections into one continuous path
    all_edges = []
    all_edge_costs = []
    all_edge_risks = []
    total_cost = 0.0
    total_risk = 0.0

    for k in range(len(sections)):
        section_path = paths.get(k, [])
        section_edges = [(section_path[i], section_path[i + 1]) for i in range(len(section_path) - 1)]
        detailed_risk_k = validated_results['detailed_risk'].get(k, {})

        for edge in section_edges:
            data = G[edge[0]][edge[1]]
            cost = data.get('cost', 0.0)
            all_edge_costs.append(cost)
            total_cost += cost

            link_risk = sum(detailed_risk_k.get(edge, {}).values())
            all_edge_risks.append(link_risk)
            total_risk += link_risk

            all_edges.append(edge)

    if not all_edges:
        print("No path available across all sections.")
        return

    # Option 1: Simple print statements
    #print("Full Path:")
    print("Edges: " + " -> ".join([f"{u}-{v}" for u, v in all_edges]) + f" (Total edges: {len(all_edges)})")
    print("Costs: " + " | ".join([f"{c:.4f}" for c in all_edge_costs]) + f" (Total: {total_cost:.4f})")
    print("Risks: " + " | ".join([f"{r:.4f}" for r in all_edge_risks]) + f" (Total: {total_risk:.4f})")
    print("-" * 50)

    # Option 2: Table for better visual alignment (recommended)
    table_data = [
        ["Edges"] + [f"{u}-{v}" for u, v in all_edges] + ["Total"],
        ["Costs"] + [f"{c:.4f}" for c in all_edge_costs] + [f"{total_cost:.4f}"],
        ["Risks"] + [f"{r:.4f}" for r in all_edge_risks] + [f"{total_risk:.4f}"]
    ]
    #print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    #print("-" * 50)



def validate_safety_constraints(
        perm: Tuple[str, ...],
        final_order: Dict[str, List[str]],
) -> Tuple[bool, List[str]]:
    """
    Args:
        perm (Tuple[str, ...]):
            : ('AA', 'BB', 'CC')

        final_order (Dict[str, List[str]]):
            : {'AA': ['F', 'A'], 'BB': ['F', 'B'], 'CC': ['F', 'F']}

        lambda_matrix (Dict[str, Dict[str, int]]):

        dist_to_engine (int):

    """

    full_train_sequence: List[str] = []
    for block_key in perm:
        cars_in_block = final_order.get(block_key)

        if cars_in_block is None:
            return False, [f"خطای داده: بلاک '{block_key}' در perm وجود دارد اما در final_order یافت نشد."]

        full_train_sequence.extend(cars_in_block)

    if not full_train_sequence:
        return False, ["خطای داده: هیچ واگنی در توالی نهایی یافت نشد."]

    is_valid: bool = True
    errors: List[str] = []
    train_len: int = len(full_train_sequence)

    num_buffer_cars_to_check = min(train_len,Dist_to_Engine)

    for i in range(num_buffer_cars_to_check):
        pos = i + 1
        car_type = full_train_sequence[i]

        if car_type != 'F':
            is_valid = False
            errors.append(f"⛔ نقض بافر: موقعیت {pos} باید 'F' (ایمن) باشد، اما '{car_type}' یافت شد.")

    for i in range(train_len - 1):
        pos1 = i + 1
        pos2 = i + 2

        t1 = full_train_sequence[i]
        t2 = full_train_sequence[i + 1]

        if LAMBDA_MATRIX.get(t1, {}).get(t2, 0) == 1:
            is_valid = False
            errors.append(f"⛔ نقض مجاورت: زوج ناسازگار ('{t1}', '{t2}') در موقعیت‌های {pos1} و {pos2} یافت شد.")

    return is_valid, errors