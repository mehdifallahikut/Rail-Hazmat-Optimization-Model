# OptPerm_LBBD_Weighted.py
# Enhanced LBBD with weighted objective (Alpha * Cost + (1-Alpha) * Risk)
# + Added Acceleration Strategies: Adaptive Initial Cuts & Multi-Cut

import time
import math
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass
import cplex
import networkx as nx
from matplotlib import pyplot as plt
from networkx import single_source_dijkstra
import tracemalloc
import psutil
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx


def draw_q1_style(G, path_nodes, title, cost_val, risk_val, time_val, is_sequential=False):
    plt.figure(figsize=(20, 10))

    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    edges = G.edges(data=True)
    for u, v, data in edges:
        exposure = data.get('exposure', 0)
        cost = data.get('cost', 0)

        # Trap: Cost ~ 45 (+/- 2) -> Max 47
        # Neutral: Cost ~ 52 (+/- 4) -> Range 48 to 56
        # Safe: Cost ~ 60 (+/- 2) -> Min 58
        if cost < 48:
            # محدوده تله (قرمز)
            color = '#e74c3c'
            width = 5.0
            alpha = 0.4
            style = 'solid'
        elif cost >= 57:
            # محدوده امن (سبز)
            color = '#2ecc71'
            width = 5.0
            alpha = 0.4
            style = 'solid'
        else:
            # محدوده خنثی (خاکستری) - بین 48 تا 57
            color = '#bdc3c7'
            width = 3.0
            alpha = 0.6  # کمی پررنگ‌تر کردم که دیده شود
            style = 'dashed'

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=width, alpha=alpha, edge_color=color, style=style)

    node_colors = []
    node_sizes = []
    labels = {}

    for n in G.nodes():
        if n == 'SS':
            node_colors.append('#2ecc71')  # سبز روشن
            node_sizes.append(650)
            labels[n] = n
        elif 'N' not in n:
            node_colors.append('#f1c40f')  # زرد طلایی
            node_sizes.append(550)
            labels[n] = n
        else:
            node_colors.append('#95a5a6')  # خاکستری
            node_sizes.append(40)

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=node_colors,
                           node_size=node_sizes, edgecolors='black', linewidths=1.5)

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

    # 3. رسم مسیر
    if path_nodes and len(path_nodes) > 1:
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        p_style = '--' if is_sequential else '-'
        p_label = 'Sequential Path' if is_sequential else 'LBBD Path (Optimal)'

        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.0,
                               edge_color='black', style=p_style, alpha=0.9)

    # 4. لجند و باکس
    handles = [
        mpatches.Patch(color='#e74c3c', label='Trap Zone (High Risk)', alpha=0.4),
        mpatches.Patch(color='#2ecc71', label='Safe Zone (High Cost)', alpha=0.4),
        mpatches.Patch(color='#bdc3c7', label='Neutral Zone', alpha=0.6, linestyle='--'),
        mlines.Line2D([], [], color='black', lw=3, linestyle=p_style, label=p_label)
    ]
    plt.legend(handles=handles, loc='upper right', frameon=True, fontsize=11)

    info_text = (f"METRICS:\n--------\nCost : {cost_val:.1f}\nRisk : {risk_val:.1f}\nTime : {time_val:.2f}s")
    plt.text(0.98, 0.76, info_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'),
             fontfamily='monospace')

    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()

    fname = "Sequential_Result_Final.png" if is_sequential else "LBBD_Result_Final.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------------
class MemoryMonitor:
    def __init__(self, section_name, active=True):  # اضافه کردن پارامتر active
        self.section_name = section_name
        self.active = active
        self.start_snapshot = None
        self.start_rss = 0

    def __enter__(self):
        if not self.active:
            return self

        # فقط اگر فعال بود، پردازش انجام شود
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self.start_rss = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.start_snapshot = tracemalloc.take_snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.active:
            return

        end_snapshot = tracemalloc.take_snapshot()
        end_rss = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        stats = end_snapshot.compare_to(self.start_snapshot, 'lineno')
        total_allocated_python = sum(stat.size_diff for stat in stats) / 1024 / 1024
        rss_diff = end_rss - self.start_rss

        print(f"\n[MEMORY REPORT] Section: '{self.section_name}'")
        print(f"   Python Allocations Diff: {total_allocated_python:.4f} MB")
        print(f"   System RSS Diff: {rss_diff:.4f} MB")
# ----------------
# Configuration Class
# ----------------
@dataclass
class BendersOptions:
    use_initial_cuts: bool = True
    initial_cuts_percentage: float = 1
    use_min_cost_cuts: bool = False
    use_min_cost_cuts2: bool = False
    debug_strategies: bool = False

# ----------------
# Type Aliases & Constants
# ----------------
BlockDict = Dict[str, List[str]]
LambdaMat = Dict[str, Dict[str, int]]
LAMBDA_MATRIX = {
    'A': {'A': 0, 'B': 1, 'C': 1, 'D': 0, 'E': 1, 'F': 0},
    'B': {'A': 1, 'B': 0, 'C': 1, 'D': 0, 'E': 1, 'F': 0},
    'C': {'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': 1, 'F': 0},
    'D': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 1, 'F': 0},
    'E': {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 0, 'F': 0},
    'F': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
}
Dist_to_Engine = 5
hazmat_types = ['A', 'B', 'C', 'D', 'E']

try:
    from data import LAMBDA_MATRIX as _LM
    DEFAULT_LAMBDA = _LM
except Exception:
    DEFAULT_LAMBDA = defaultdict(lambda: defaultdict(int))
import random


def solve_subproblem_for_k(
        G: nx.DiGraph,
        verbose: bool,
        start: str,
        end: str,
        y_k: List[float],
        L_k: int,
        alpha: float = 0.5
) -> Tuple[float, float, List[Tuple[str, str]], List[str], Dict[Tuple[str, str], Dict[int, float]], Dict[str, float]]:
    if L_k <= 0:
        return 0.0, 0.0, [], [start, end] if start != end else [], {}, {}

    if len(y_k) < L_k:
        y_k = y_k + [0.0] * (L_k - len(y_k))
    else:
        y_k = y_k[:L_k]

    epsilon = 1e-6

    def weight(u, v, d):
        P = d.get('P', [0.0] * L_k)
        exp = float(d.get('exposure', 0.0))
        up = min(len(P), L_k)
        risk_edge = sum(y_k[p] * float(P[p]) for p in range(up) if y_k[p] > 0)
        cost_val = float(d.get('cost', 0.0))
        total_risk = exp * risk_edge
        real_val = (alpha * cost_val) + ((1.0 - alpha) * total_risk)
        noise = random.uniform(0, epsilon)
        return real_val + noise

    try:
        lengths, paths = single_source_dijkstra(G, start, weight=weight)
        if end not in lengths:
            raise nx.NetworkXNoPath
        path_nodes = paths[end]
        if not path_nodes or path_nodes[0] != start:
            raise nx.NetworkXNoPath
    except nx.NetworkXNoPath:
        if verbose:
            print(f"DEBUG: No path for {start} -> {end}")
        return float('inf'), float('inf'), [], [], {}, {}

    duals = {node: lengths.get(node, float('inf')) for node in G.nodes}
    used_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]


    cost_val = 0.0
    risk_val = 0.0
    detailed_risk: Dict[Tuple[str, str], Dict[int, float]] = {}

    for u, v in used_edges:
        d = G.edges[u, v]
        c = float(d.get('cost', 0.0))
        exp = float(d.get('exposure', 0.0))
        P = d.get('P', [0.0] * L_k)

        cost_val += c

        edge_risk = 0.0
        detailed_risk[(u, v)] = {}
        up = min(len(P), L_k)
        for p in range(up):
            if y_k[p] > 0:
                contrib = y_k[p] * exp * float(P[p])
                detailed_risk[(u, v)][p + 1] = contrib
                edge_risk += contrib
        risk_val += edge_risk

    if verbose:
        print(f"SP path {start}->{end}: cost={cost_val:.4f}, risk={risk_val:.4f}")

    return cost_val, risk_val, used_edges, path_nodes, detailed_risk, duals

def compute_bigM_for_cuts(
        G: nx.DiGraph,
        key_nodes: List[str],
        train_length: int,
        bounds: Dict,
        alpha: float = 0.5,
        debug: bool = False
) -> Tuple[float, List[float], Dict[Tuple[str, str], float]]:
    M_p = bounds['max_risk_per_pos']
    C_ub_val = bounds['global_max_cost']
    risk_global = bounds['global_max_risk']

    M_pair: Dict[Tuple[str, str], float] = {}
    for s in key_nodes:
        for t in key_nodes:
            if s == t:
                continue
            R_ub_st = risk_global
            M_pair[(s, t)] = float((alpha * C_ub_val) + ((1.0 - alpha) * R_ub_st))
    M_theta = max(M_pair.values()) if M_pair else float((alpha * C_ub_val) + ((1.0 - alpha) * risk_global))

    if debug:
        print(f"Big-M: M_theta={M_theta:.6f}, risk_global={risk_global:.6f}, alpha={alpha}")

    return M_theta, M_p, M_pair



def build_master(
        block_keys: List[str],
        blocks: BlockDict,
        train_length: int,
        G: nx.DiGraph,
        bounds: Dict,
        verbose: bool,
        alpha: float = 0.5
):
    model = cplex.Cplex()
    model.set_problem_type(cplex.Cplex.problem_type.MILP)
    model.objective.set_sense(model.objective.sense.minimize)
    model.parameters.mip.tolerances.mipgap.set(1e-6)
    model.set_results_stream(None)
    model.set_log_stream(None)
    model.set_warning_stream(None)
    #model.parameters.threads.set(1)

    nodes = list(G.nodes)
    num_sections = len(block_keys)
    L = train_length

    car_counts_per_block: Dict[str, Dict[str, int]] = {}
    for b in block_keys:
        car_counts_per_block[b] = defaultdict(int)
        for t in blocks[b]:
            car_counts_per_block[b][t] += 1

    types = sorted({t for b in blocks for t in blocks[b]})
    block_lengths = {b: len(blocks[b]) for b in block_keys}
    key_nodes = ['SS'] + block_keys

    # X[b,t,pos]
    X = {}
    x_names, x_types = [], []
    x_ub_list = []
    for b in block_keys:
        for t in car_counts_per_block[b].keys():
            for pos in range(1, L + 1):
                nm = f'X_{b}_{t}_{pos}'
                X[b, t, pos] = nm
                x_names.append(nm)
                x_types.append(model.variables.type.binary)
                if pos <= Dist_to_Engine and t != 'F':
                    x_ub_list.append((nm, 0.0))

    model.variables.add(names=x_names, types=x_types)
    if x_ub_list:
        model.variables.set_upper_bounds(x_ub_list)

    # Car count constraints
    car_lin, car_sense, car_rhs = [], [], []
    for b in block_keys:
        for t in car_counts_per_block[b].keys():
            vars_car = [X[b, t, pos] for pos in range(1, L + 1)]
            car_lin.append([vars_car, [1.0] * len(vars_car)])
            car_sense.append("E")
            car_rhs.append(float(car_counts_per_block[b][t]))
    model.linear_constraints.add(lin_expr=car_lin, senses=car_sense, rhs=car_rhs)

    # z[b,pos] and u[b,q]
    z = {}
    z_names, z_types = [], []
    for b in block_keys:
        for pos in range(1, L + 1):
            nm = f'z_{b}_{pos}'
            z[b, pos] = nm
            z_names.append(nm)
            z_types.append(model.variables.type.binary)
    model.variables.add(names=z_names, types=z_types)

    u = {}
    u_names, u_types = [], []
    for b in block_keys:
        for q in range(1, L - block_lengths[b] + 2):
            nm = f"u_{b}_{q}"
            u[b, q] = nm
            u_names.append(nm)
            u_types.append(model.variables.type.binary)
    model.variables.add(names=u_names, types=u_types)

    # One u per block
    u_sum_lin, u_sum_sense, u_sum_rhs = [], [], []
    for b in block_keys:
        u_sum_lin.append([[u[b, q] for q in range(1, L - block_lengths[b] + 2)],
                          [1.0] * (L - block_lengths[b] + 1)])
        u_sum_sense.append("E")
        u_sum_rhs.append(1.0)
    model.linear_constraints.add(lin_expr=u_sum_lin, senses=u_sum_sense, rhs=u_sum_rhs)

    # Link z to u
    z_u_lin, z_u_sense, z_u_rhs = [], [], []
    for b in block_keys:
        lenb = block_lengths[b]
        for pos in range(1, L + 1):
            cover_q = [q for q in range(1, L - lenb + 2) if q <= pos <= q + lenb - 1]
            if not cover_q:
                z_u_lin.append([[z[b, pos]], [1.0]])
                z_u_sense.append("E")
                z_u_rhs.append(0.0)
            else:
                z_u_lin.append([[z[b, pos]] + [u[b, q] for q in cover_q], [1.0] + [-1.0] * len(cover_q)])
                z_u_sense.append("E")
                z_u_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=z_u_lin, senses=z_u_sense, rhs=z_u_rhs)

    # Link z to X
    z_x_lin, z_x_sense, z_x_rhs = [], [], []
    for b in block_keys:
        for pos in range(1, L + 1):
            vars_z = [X[b, t, pos] for t in car_counts_per_block[b].keys()]
            z_x_lin.append([[z[b, pos]] + vars_z, [-1.0] + [1.0] * len(vars_z)])
            z_x_sense.append("E")
            z_x_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=z_x_lin, senses=z_x_sense, rhs=z_x_rhs)

    # Block length constraints
    cnt_lin, cnt_sense, cnt_rhs = [], [], []
    for b in block_keys:
        cnt_lin.append([[z[b, pos] for pos in range(1, L + 1)], [1.0] * L])
        cnt_sense.append("E")
        cnt_rhs.append(float(block_lengths[b]))
    model.linear_constraints.add(lin_expr=cnt_lin, senses=cnt_sense, rhs=cnt_rhs)

    # s[pos,t]
    s = {}
    s_names, s_types, s_bounds = [], [], []
    for pos in range(1, L + 1):
        for t in types:
            nm = f's_{pos}_{t}'
            s[pos, t] = nm
            s_names.append(nm)
            s_types.append(model.variables.type.continuous)
            s_bounds.append((nm, 0.0))
            s_bounds.append((nm, 1.0))
    model.variables.add(names=s_names, types=s_types)
    model.variables.set_lower_bounds(s_bounds[::2])
    model.variables.set_upper_bounds(s_bounds[1::2])

    # Link s to X
    s_lin, s_sense, s_rhs = [], [], []
    for pos in range(1, L + 1):
        for t in types:
            vars_s = [X[b, t, pos] for b in block_keys if t in car_counts_per_block[b]]
            s_lin.append([[s[pos, t]] + vars_s, [-1.0] + [1.0] * len(vars_s)])
            s_sense.append("E")
            s_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=s_lin, senses=s_sense, rhs=s_rhs)

    # y[pos]
    y = {}
    y_names, y_types = [], []
    for pos in range(1, L + 1):
        nm = f'y_{pos}'
        y[pos] = nm
        y_names.append(nm)
        y_types.append(model.variables.type.binary)
    model.variables.add(names=y_names, types=y_types)

    # y + s[F] = 1
    y_lin, y_sense, y_rhs = [], [], []
    for pos in range(1, L + 1):
        y_lin.append([[y[pos], s[pos, 'F']], [1.0, 1.0]])
        y_sense.append("E")
        y_rhs.append(1.0)
    model.linear_constraints.add(lin_expr=y_lin, senses=y_sense, rhs=y_rhs)

    # Incompatibility constraints
    incomp_lin, incomp_sense, incomp_rhs = [], [], []
    for pos in range(1, L):
        pos2 = pos + 1
        for t1 in types:
            for t2 in types:
                if LAMBDA_MATRIX.get(t1, {}).get(t2, 0) == 1:
                    incomp_lin.append([[s[pos, t1], s[pos2, t2]], [1.0, 1.0]])
                    incomp_sense.append("L")
                    incomp_rhs.append(1.0)
    if incomp_lin:
        model.linear_constraints.add(lin_expr=incomp_lin, senses=incomp_sense, rhs=incomp_rhs)

    # r[k,b]
    r = {}
    r_names, r_types = [], []
    for k in range(1, num_sections + 1):
        for b in block_keys:
            nm = f'r_{k}_{b}'
            r[k, b] = nm
            r_names.append(nm)
            r_types.append(model.variables.type.binary)
    model.variables.add(names=r_names, types=r_types)

    # Each block to one rank
    r_b_lin, r_b_sense, r_b_rhs = [], [], []
    for b in block_keys:
        r_b_lin.append([[r[k, b] for k in range(1, num_sections + 1)], [1.0] * num_sections])
        r_b_sense.append("E")
        r_b_rhs.append(1.0)
    model.linear_constraints.add(lin_expr=r_b_lin, senses=r_b_sense, rhs=r_b_rhs)

    # Each rank to one block
    r_k_lin, r_k_sense, r_k_rhs = [], [], []
    for k in range(1, num_sections + 1):
        r_k_lin.append([[r[k, b] for b in block_keys], [1.0] * len(block_keys)])
        r_k_sense.append("E")
        r_k_rhs.append(1.0)
    model.linear_constraints.add(lin_expr=r_k_lin, senses=r_k_sense, rhs=r_k_rhs)

    # min_pos_u
    min_pos_u = {}
    minpos_names, minpos_types, minpos_bounds = [], [], []
    for b in block_keys:
        nm = f'min_pos_u_{b}'
        min_pos_u[b] = nm
        minpos_names.append(nm)
        minpos_types.append(model.variables.type.integer)
        minpos_bounds.append((nm, 1.0))
        minpos_bounds.append((nm, float(L - block_lengths[b] + 1)))
    model.variables.add(names=minpos_names, types=minpos_types)
    model.variables.set_lower_bounds(minpos_bounds[::2])
    model.variables.set_upper_bounds(minpos_bounds[1::2])

    # Link min_pos_u to u
    minpos_u_lin, minpos_u_sense, minpos_u_rhs = [], [], []
    for b in block_keys:
        vars_u = [u[b, q] for q in range(1, L - block_lengths[b] + 2)]
        coeffs_u = [float(q) for q in range(1, L - block_lengths[b] + 2)]
        minpos_u_lin.append([[min_pos_u[b]] + vars_u, [-1.0] + coeffs_u])
        minpos_u_sense.append("E")
        minpos_u_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=minpos_u_lin, senses=minpos_u_sense, rhs=minpos_u_rhs)

    # Engine distance constraint
    for b in block_keys:
        safe_count = car_counts_per_block[b].get('F', 0)
        if safe_count < block_lengths[b]:
            min_start_pos = max(1.0, float(6 - safe_count))
            if min_start_pos > 1.0:
                model.variables.set_lower_bounds([(min_pos_u[b], min_start_pos)])

    # start_k
    start_k = {}
    sk_names, sk_types, sk_bounds = [], [], []
    min_block_len = min(block_lengths.values()) if block_lengths else 1
    max_block_len = max(block_lengths.values()) if block_lengths else train_length

    for k in range(1, num_sections + 1):
        nm = f'start_k_{k}'
        start_k[k] = nm
        sk_names.append(nm)
        sk_types.append(model.variables.type.integer)
        min_start = min_block_len * (num_sections - k) + 1
        max_start_1 = train_length - min_block_len * (k - 1)
        max_start_2 = 1 + (num_sections - k) * max_block_len
        max_start = min(max_start_1, max_start_2)
        sk_bounds.append((nm, max(1.0, float(min_start))))
        sk_bounds.append((nm, min(float(train_length), float(max_start))))
    model.variables.add(names=sk_names, types=sk_types)
    model.variables.set_lower_bounds(sk_bounds[::2])
    model.variables.set_upper_bounds(sk_bounds[1::2])

    # Link start_k to min_pos_u via r
    sk_lin, sk_sense, sk_rhs = [], [], []
    M_pos_upper = float(train_length - 1)
    for k in range(1, num_sections + 1):
        for b in block_keys:
            M_pos_lower = float(train_length - block_lengths[b])
            sk_lin.append([[start_k[k], min_pos_u[b], r[k, b]], [1.0, -1.0, M_pos_upper]])
            sk_sense.append("L")
            sk_rhs.append(M_pos_upper)
            sk_lin.append([[start_k[k], min_pos_u[b], r[k, b]], [1.0, -1.0, -M_pos_lower]])
            sk_sense.append("G")
            sk_rhs.append(-M_pos_lower)
    model.linear_constraints.add(lin_expr=sk_lin, senses=sk_sense, rhs=sk_rhs)

    # Sink/source chain
    is_sink = {}
    sink_names, sink_types, sink_ub = [], [], []
    for k in range(num_sections):
        for node in nodes:
            nm = f'is_sink_{k}_{node}'
            is_sink[k, node] = nm
            sink_names.append(nm)
            sink_types.append(model.variables.type.binary)
            if node not in block_keys:
                sink_ub.append((nm, 0.0))
    model.variables.add(names=sink_names, types=sink_types)
    if sink_ub:
        model.variables.set_upper_bounds(sink_ub)

    # Link sink to r
    sink_lin, sink_sense, sink_rhs = [], [], []
    for k in range(num_sections):
        for b in block_keys:
            sink_lin.append([[is_sink[k, b], r[k + 1, b]], [1.0, -1.0]])
            sink_sense.append("E")
            sink_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=sink_lin, senses=sink_sense, rhs=sink_rhs)

    is_source = {}
    src_names, src_types, src_bounds = [], [], []
    for k in range(num_sections):
        for node in nodes:
            nm = f'is_source_{k}_{node}'
            is_source[k, node] = nm
            src_names.append(nm)
            src_types.append(model.variables.type.binary)
    model.variables.add(names=src_names, types=src_types)

    # Fix first source to SS
    src_fix = []
    for node in nodes:
        nm = is_source[0, node]
        if node == 'SS':
            src_fix.append((nm, 1.0))
        else:
            src_fix.append((nm, 0.0))
    model.variables.set_lower_bounds(src_fix)
    model.variables.set_upper_bounds(src_fix)

    # Link source to sink
    srcsink_lin, srcsink_sense, srcsink_rhs = [], [], []
    for k in range(1, num_sections):
        for b in block_keys:
            srcsink_lin.append([[is_source[k, b], is_sink[k - 1, b]], [1.0, -1.0]])
            srcsink_sense.append("E")
            srcsink_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=srcsink_lin, senses=srcsink_sense, rhs=srcsink_rhs)

    # lengths[k]
    lengths = {}
    Lk_names, Lk_types, Lk_bounds = [], [], []
    for k in range(num_sections):
        nm = f'L_{k}'
        lengths[k] = nm
        Lk_names.append(nm)
        Lk_types.append(model.variables.type.integer)
        num_remaining_blocks = num_sections - k
        min_L_k = float(num_remaining_blocks * min_block_len)
        max_L_k = float(num_remaining_blocks * max_block_len)
        Lk_bounds.append((nm, min_L_k))
        Lk_bounds.append((nm, max_L_k))
    model.variables.add(names=Lk_names, types=Lk_types)
    model.variables.set_lower_bounds(Lk_bounds[::2])
    model.variables.set_upper_bounds(Lk_bounds[1::2])

    # Link start_k to lengths
    start_len_lin, start_len_sense, start_len_rhs = [], [], []
    for k in range(1, num_sections):
        start_len_lin.append([[start_k[k], lengths[k]], [1.0, -1.0]])
        start_len_sense.append("E")
        start_len_rhs.append(1.0)
    start_len_lin.append([[start_k[num_sections]], [1.0]])
    start_len_sense.append("E")
    start_len_rhs.append(1.0)
    model.linear_constraints.add(lin_expr=start_len_lin, senses=start_len_sense, rhs=start_len_rhs)

    # len_j per block
    len_j = {}
    lj_names, lj_types = [], []
    for j in range(1, num_sections + 1):
        nm = f'len_j_{j}'
        len_j[j] = nm
        lj_names.append(nm)
        lj_types.append(model.variables.type.integer)
    model.variables.add(names=lj_names, types=lj_types)

    lj_lin, lj_sense, lj_rhs = [], [], []
    for j in range(1, num_sections + 1):
        vars_len = [r[j, b] for b in block_keys]
        lj_lin.append([[len_j[j]] + vars_len, [-1.0] + [float(block_lengths[b]) for b in block_keys]])
        lj_sense.append("E")
        lj_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=lj_lin, senses=lj_sense, rhs=lj_rhs)

    # Sum lengths
    Lk_lin, Lk_sense, Lk_rhs = [], [], []
    for k in range(num_sections):
        vars_l = [len_j[j] for j in range(k + 1, num_sections + 1)]
        Lk_lin.append([[lengths[k]] + vars_l, [-1.0] + [1.0] * len(vars_l)])
        Lk_sense.append("E")
        Lk_rhs.append(0.0)
    model.linear_constraints.add(lin_expr=Lk_lin, senses=Lk_sense, rhs=Lk_rhs)

    # is_active[k,pos]
    is_active = {}
    ia_names, ia_types = [], []
    for k in range(num_sections):
        for pos in range(1, L + 1):
            nm = f'is_active_{k}_{pos}'
            is_active[k, pos] = nm
            ia_names.append(nm)
            ia_types.append(model.variables.type.binary)
    model.variables.add(names=ia_names, types=ia_types)

    ia_lin, ia_sense, ia_rhs = [], [], []
    for k in range(num_sections):
        for pos in range(1, L + 1):
            ia_lin.append([[lengths[k], is_active[k, pos]], [1.0, -L]])
            ia_sense.append("G")
            ia_rhs.append(float(pos - L))
            ia_lin.append([[lengths[k], is_active[k, pos]], [1.0, -L]])
            ia_sense.append("L")
            ia_rhs.append(float(pos - 1))
    model.linear_constraints.add(lin_expr=ia_lin, senses=ia_sense, rhs=ia_rhs)

    # w[k,pos] gating
    w = {}
    w_names, w_types = [], []
    for k in range(num_sections):
        for pos in range(1, L + 1):
            nm = f'w_{k}_{pos}'
            w[k, pos] = nm
            w_names.append(nm)
            w_types.append(model.variables.type.binary)
    model.variables.add(names=w_names, types=w_types)

    w_lin, w_sense, w_rhs = [], [], []
    for k in range(num_sections):
        for pos in range(1, L + 1):
            wv, yv, av = w[k, pos], y[pos], is_active[k, pos]
            w_lin.append([[wv, yv], [1.0, -1.0]])
            w_sense.append("L")
            w_rhs.append(0.0)
            w_lin.append([[wv, av], [1.0, -1.0]])
            w_sense.append("L")
            w_rhs.append(0.0)
            w_lin.append([[wv, yv, av], [1.0, -1.0, -1.0]])
            w_sense.append("G")
            w_rhs.append(-1.0)
    model.linear_constraints.add(lin_expr=w_lin, senses=w_sense, rhs=w_rhs)

    # theta[k] variables
    theta = {}
    theta_names, theta_types, theta_bounds = [], [], []
    for k in range(num_sections):
        nm = f'theta_{k}'
        theta[k] = nm
        theta_names.append(nm)
        theta_types.append(model.variables.type.continuous)

        theta_bounds.append((nm, 0.0))
        max_val = (alpha * bounds['global_max_cost']) + ((1.0 - alpha) * bounds['global_max_risk'])
        theta_bounds.append((nm, max_val))

    model.variables.add(names=theta_names, types=theta_types)
    model.variables.set_lower_bounds(theta_bounds[::2])
    model.variables.set_upper_bounds(theta_bounds[1::2])

    # Objective
    model.objective.set_linear([(theta[k], 1.0) for k in range(num_sections)])

    # aux[k,s,t] for gated cuts
    aux = {}
    aux_lin, aux_sense, aux_rhs = [], [], []
    for k in range(num_sections):
        for s_node in key_nodes:
            for t_node in block_keys:
                if s_node == t_node:
                    continue
                nm = f'aux_{k}_{s_node}_{t_node}'
                aux[k, s_node, t_node] = nm
                model.variables.add(names=[nm], types=[model.variables.type.binary])
                src_var = is_source[k, s_node]
                sink_var = is_sink[k, t_node]
                aux_lin.append([[nm, src_var], [1.0, -1.0]])
                aux_sense.append("L")
                aux_rhs.append(0.0)
                aux_lin.append([[nm, sink_var], [1.0, -1.0]])
                aux_sense.append("L")
                aux_rhs.append(0.0)
                aux_lin.append([[nm, src_var, sink_var], [1.0, -1.0, -1.0]])
                aux_sense.append("G")
                aux_rhs.append(-1.0)
    model.linear_constraints.add(lin_expr=aux_lin, senses=aux_sense, rhs=aux_rhs)

    if verbose:
        print(f"Master: Vars={model.variables.get_num()}, Cons={model.linear_constraints.get_num()}, Alpha={alpha}")

    return {
        'model': model,
        'X': X,
        'z': z,
        's': s,
        'y': y,
        'r': r,
        'w': w,
        'len_j': len_j,
        'lengths': lengths,
        'is_source': is_source,
        'is_sink': is_sink,
        'is_active': is_active,
        'theta': theta,
        'types': types,
        'block_lengths': block_lengths,
        'num_sections': num_sections,
        'train_length': L,
        'aux': aux,
        'key_nodes': key_nodes
    }


def get_final_order(X, solution_values: Dict[str, float], block_keys: List[str]) -> Dict[str, List[str]]:
    final_order: Dict[str, List[str]] = {b: [] for b in block_keys}
    assigned = []
    for (b, t, pos), nm in X.items():
        if solution_values.get(nm, 0.0) > 0.5:
            assigned.append((b, t, pos))
    for b in block_keys:
        block_ass = sorted([(t, pos) for bb, t, pos in assigned if bb == b], key=lambda x: x[1])
        final_order[b] = [t for t, _ in block_ass]
    return final_order


def evaluate_final_solution_from_y(
        G,
        y_vars,
        L,
        sections,
        lengths,
        sol_values,
        alpha: float = 0.5,
        debug=False
):
    tol = 1e-9
    y_bin = [1.0 if sol_values.get(y_vars[pos], 0.0) >= 1.0 - tol else 0.0
             for pos in range(1, L + 1)]

    total_cost = 0.0
    total_risk = 0.0
    risk_details = {}
    paths = {}

    for k, (s_node, t_node) in enumerate(sections):
        L_k = max(0, int(round(lengths[k])))
        y_k = y_bin[:L_k] if L_k <= len(y_bin) else y_bin + [0.0] * (L_k - len(y_bin))
        c_k, r_k, _, path_nodes, det_risk_k, _ = solve_subproblem_for_k(G, debug, s_node, t_node, y_k, L_k, alpha=alpha)
        total_cost += c_k
        total_risk += r_k
        risk_details[k] = det_risk_k
        paths[k] = path_nodes
        if debug:
            print(f"Section {k}: {s_node}->{t_node}, L={L_k}, cost={c_k:.6f}, risk={r_k:.6f}")

    return total_cost, total_risk, risk_details, paths


def run_optPerm_LBBD_Weighted(
        block_keys: List[str],
        blocks: BlockDict,
        train_length: int,
        G: nx.DiGraph,
        timeout: int = 0,
        alpha: float = 0.5,
        debug: bool = False,
        options: BendersOptions = BendersOptions()  # Default options added
):
    if not block_keys:
        return None
    stats_history = []
    solve_start = time.time()
    key_nodes = ['SS'] + block_keys
    with MemoryMonitor("Weighted LBBD: Precompute Bounds",active=options.debug_strategies):
        bounds = precompute_bounds(G, key_nodes, train_length)
        min_risk_dict = bounds['min_risk_dict']
        min_cost_dict = bounds['new_min_cost_dict']

    res = build_master(block_keys, blocks, train_length, G, bounds, verbose=debug, alpha=alpha)
    model = res['model']
    aux = res['aux']
    with MemoryMonitor("Weighted LBBD: Big-M Calculation",active=options.debug_strategies):
        M_theta, M_p, M_pair = compute_bigM_for_cuts(G, res['key_nodes'], train_length, bounds, alpha=alpha, debug=debug)

    # ------------------------------------------------------------
    # OPTIONAL STRATEGY: Min-Cost Valid Inequalities
    # ------------------------------------------------------------

    if options.use_min_cost_cuts:
        with MemoryMonitor("Weighted LBBD: Min-Cost Cuts",active=options.debug_strategies):
            if debug: print("Adding Min-Cost initial constraints...")
            count_min_cost = 0
            for k in range(res['num_sections']):
                for s_node in block_keys + ['SS']:
                    for t_node in block_keys:
                        if s_node == t_node: continue
                        min_c = min_cost_dict.get(s_node, {}).get(t_node, 0.0)
                        if min_c > 0:
                            # Theta[k] >= alpha * min_c * aux[k, s, t]
                            aux_var = res['aux'][k, s_node, t_node]
                            model.linear_constraints.add(
                                lin_expr=[[[res['theta'][k], aux_var], [1.0, -alpha * min_c]]],
                                senses=["G"],
                                rhs=[0.0]
                            )
                            count_min_cost+=1

            stats_history.append(
                {'time': time.time() - solve_start, 'type': 'Init_MinCost', 'count': count_min_cost})
    if options.use_min_cost_cuts2:
        with MemoryMonitor("Weighted LBBD: Strong Objective Cuts",active=options.debug_strategies):
            if debug: print("Adding Strong Objective (Cost+Risk) Cuts...")
            count_strong = 0
            for k in range(res['num_sections']):
                for s_node in block_keys + ['SS']:
                    for t_node in block_keys:
                        if s_node == t_node: continue

                        min_c = min_cost_dict.get(s_node, {}).get(t_node, 0.0)

                        M_st = M_pair.get((s_node, t_node), M_theta)

                        has_risk = False
                        for pos in range(1, train_length + 1):
                            if min_risk_dict.get(pos, {}).get(s_node, {}).get(t_node, 0.0) > 1e-9:
                                has_risk = True
                                break

                        if min_c > 0 or has_risk:
                            aux_var = res['aux'][k, s_node, t_node]
                            # Theta - Sum(Risk_term * w) - M * Aux >= alpha * Cost - M
                            gen_vars = [res['theta'][k], aux_var]
                            gen_coeff = [1.0, -M_st]

                            rhs_val = (alpha * min_c) - M_st

                            for pos in range(1, train_length + 1):
                                r_val = min_risk_dict.get(pos, {}).get(s_node, {}).get(t_node, 0.0)
                                if r_val > 1e-12:
                                    # Theta >= ... + coeff * w  => Theta - coeff * w >= ...
                                    term = (1.0 - alpha) * r_val
                                    gen_vars.append(res['w'][k, pos])
                                    gen_coeff.append(-term)

                            model.linear_constraints.add(
                                lin_expr=[[gen_vars, gen_coeff]],
                                senses=["G"],
                                rhs=[rhs_val]
                            )
                            count_strong+=1
        if count_strong > 0:
            stats_history.append({'time': time.time() - solve_start, 'type': 'Init_StrongObj', 'count': count_strong})
    # ==============================================================================
    # ------------------------------------------------------------
    # STRATEGY 1: ADAPTIVE INITIAL CUTS
    # ------------------------------------------------------------
    if options.use_initial_cuts:
        with MemoryMonitor("Weighted LBBD: Adaptive Initial Cuts",active=options.debug_strategies):
            potential_cuts = []
            for k in range(res['num_sections']):
                for s_node in res['key_nodes']:
                    for t_node in block_keys:
                        if s_node == t_node: continue

                        minc = min_cost_dict.get(s_node, {}).get(t_node, float('inf'))
                        if minc == float('inf'): continue

                        M_st = M_pair.get((s_node, t_node), M_theta)
                        # Heuristic strength: Higher alpha*cost means tighter lower bound on theta
                        # strength = alpha * minc
                        minr = sum(min_risk_dict.get(pos, {}).get(s_node, {}).get(t_node, 0.0)
                                   for pos in range(1, train_length + 1)) / train_length
                        strength = (alpha * minc) + ((1.0 - alpha) * minr)
                        potential_cuts.append({
                            'k': k, 's': s_node, 't': t_node,
                            'minc': minc, 'M_st': M_st, 'strength': strength
                        })

            # Sort by strength (descending)
            potential_cuts.sort(key=lambda x: x['strength'], reverse=True)
            # Select top percentage
            count_to_add = int(len(potential_cuts) * options.initial_cuts_percentage)
            selected_cuts = potential_cuts[:max(1, count_to_add)]

            if debug or options.debug_strategies:
                print(
                    f"Adding {len(selected_cuts)} adaptive initial cuts (Top {options.initial_cuts_percentage * 100}%)")

            cuts_lin, cuts_sense, cuts_rhs = [], [], []
            for cut in selected_cuts:
                k, s, t, minc, M_st = cut['k'], cut['s'], cut['t'], cut['minc'], cut['M_st']
                aux_var = aux[k, s, t]

                gen_vars = [res['theta'][k], aux_var]
                rhs_val_for_coeff = (alpha * minc) + M_st
                gen_coeff = [1.0, -rhs_val_for_coeff]

                for pos in range(1, train_length + 1):
                    c = min_risk_dict.get(pos, {}).get(s, {}).get(t, 0.0)
                    if abs(c) > 1e-12:
                        gen_vars.append(res['w'][k, pos])
                        gen_coeff.append(-(1.0 - alpha) * c)

                cuts_lin.append([gen_vars, gen_coeff])
                cuts_sense.append("G")
                cuts_rhs.append(-M_st)

            if cuts_lin:
                model.linear_constraints.add(lin_expr=cuts_lin, senses=cuts_sense, rhs=cuts_rhs)
                stats_history.append(
                    {'time': time.time() - solve_start, 'type': 'Init_Adaptive', 'count': len(cuts_lin)})

    # 4. Main Benders Loop
    max_iterations = 100000
    best_obj = float('inf')
    best_sol_values: Dict[str, float] = {}
    convergence_history = []
    final_lower_bound = -float('inf')
    epsilon = 1e-9
    previous_lb = -float('inf')
    stagnation_count = 0
    seen_sink_patterns: Dict[Tuple[str, ...], int] = {}
    no_good_limit = 3

    def add_objective_cutoff(UB_val: float):
        model.linear_constraints.add(
            lin_expr=[[[res['theta'][k] for k in range(res['num_sections'])], [1.0] * res['num_sections']]],
            senses=["L"], rhs=[UB_val]
        )

    for it in range(1, max_iterations + 1):
        current_time = time.time()
        elapsed = current_time - solve_start
        remaining = timeout - elapsed

        if remaining <= 0:
            if debug: print(f"\n[Global Timeout] Time limit reached ({timeout}s).")
            break

        if debug: print(f"Iter {it} (Rem: {remaining:.1f}s): ", end="")

        model.parameters.timelimit.set(max(1.0, remaining))
        model.solve()

        status = model.solution.get_status()
        if status not in [101, 102, 107, 109]:
            if debug: print(f"Master stop/infeasible status={status}")
            break

        names = model.variables.get_names()
        vals = model.solution.get_values(names)
        sol_values = dict(zip(names, vals))

        lower_bound = float(model.solution.get_objective_value())
        final_lower_bound = lower_bound

        if debug: print(f"LB={lower_bound:.4f}", end="")

        if abs(lower_bound - previous_lb) < epsilon:
            stagnation_count += 1
            if stagnation_count >= 10 and math.isfinite(best_obj):
                add_objective_cutoff(best_obj - 1e-9)
                stagnation_count = 0
        else:
            stagnation_count = 0
        previous_lb = lower_bound

        # Subproblem Generation
        drop_order = []
        for k in range(res['num_sections']):
            end_node = None
            for node in block_keys:
                if sol_values.get(res['is_sink'][k, node], 0.0) > 0.5:
                    end_node = node
                    break
            drop_order.append(end_node)

        sinks_tuple = tuple(drop_order)

        theta_vals = [sol_values.get(res['theta'][k], 0.0) for k in range(res['num_sections'])]
        y_values = [sol_values.get(res['y'][pos], 0.0) for pos in range(1, train_length + 1)]
        y_bin = [1 if v > 0.5 else 0 for v in y_values]
        lengths_vals = [int(round(sol_values.get(res['lengths'][k], 0))) for k in range(res['num_sections'])]

        sections = []
        start_node = 'SS'
        for k in range(res['num_sections']):
            end_node = drop_order[k]
            sections.append((start_node, end_node))
            if end_node: start_node = end_node
        if None not in sinks_tuple:
            c = seen_sink_patterns.get(sinks_tuple, 0) + 1
            seen_sink_patterns[sinks_tuple] = c
            if c >= no_good_limit and math.isfinite(best_obj) and lower_bound >= best_obj - 1e-9:
                ng_vars = [res['is_sink'][k, sinks_tuple[k]] for k in range(res['num_sections'])]
                model.linear_constraints.add(lin_expr=[[ng_vars, [1.0] * len(ng_vars)]], senses=["L"],rhs=[len(ng_vars) - 1])
                stats_history.append({'time': time.time() - solve_start, 'type': 'Benders_NoGood', 'count': 1})
        total_cost = 0.0
        total_risk = 0.0
        feasible_all = True
        cuts_added = 0

        # --------------------------------------------------------
        it_opt_cuts = 0
        it_feas_cuts = 0
        for k, (s_k, t_k) in enumerate(sections):
            L_k = max(0, lengths_vals[k])

            if L_k <= 0 or s_k is None or t_k is None:
                feasible_all = False
                if s_k and t_k:
                    model.linear_constraints.add(
                        lin_expr=[[[res['is_source'][k, s_k], res['is_sink'][k, t_k]], [1.0, 1.0]]],
                        senses=["L"], rhs=[1.0])
                    cuts_added += 1
                it_feas_cuts += 1
                continue

            y_k = y_bin[:L_k] if len(y_bin) >= L_k else y_bin + [0] * (L_k - len(y_bin))

            cost_k, risk_k, used_edges, _, _, _ = solve_subproblem_for_k(
                G, debug, s_k, t_k, y_k, L_k, alpha=alpha
            )

            if cost_k == float('inf'):
                feasible_all = False
                model.linear_constraints.add(
                    lin_expr=[[[res['is_source'][k, s_k], res['is_sink'][k, t_k]], [1.0, 1.0]]],
                    senses=["L"], rhs=[1.0])
                cuts_added += 1
                it_feas_cuts += 1
                continue

            total_cost += cost_k
            total_risk += risk_k

            alpha_k_coeffs = [0.0] * train_length
            for pos in range(1, L_k + 1):
                a = 0.0
                for (u, v) in used_edges:
                    d = G.edges[u, v]
                    exp = float(d.get('exposure', 0.0))
                    P = d.get('P', [0.0] * L_k)
                    if pos - 1 < len(P):
                        a += exp * float(P[pos - 1])
                alpha_k_coeffs[pos - 1] = a

            real_obj_k = (alpha * cost_k) + ((1.0 - alpha) * risk_k)

            if real_obj_k - theta_vals[k] > epsilon:
                aux_var = res['aux'][k, s_k, t_k]
                M_st = M_pair.get((s_k, t_k), M_theta)
                gen_vars = [res['theta'][k], aux_var]
                gen_coeff = [1.0, -M_st]
                rhs_val = (alpha * cost_k) - M_st

                for pos in range(1, L_k + 1):
                    risk_factor = alpha_k_coeffs[pos - 1]
                    if abs(risk_factor) > 1e-12:
                        gen_vars.append(res['w'][k, pos])
                        gen_coeff.append(-(1.0 - alpha) * risk_factor)

                model.linear_constraints.add(
                    lin_expr=[[gen_vars, gen_coeff]],
                    senses=["G"],
                    rhs=[rhs_val])
                stats_history.append({'time': time.time() - solve_start, 'type': 'Benders_Optimality', 'count': 1})
                cuts_added += 1
                it_opt_cuts += 1
        current_ub = (alpha * total_cost) + ((1.0 - alpha) * total_risk) if feasible_all else float('inf')
        current_ts = time.time() - solve_start
        if it_opt_cuts > 0:
            stats_history.append({'time': current_ts, 'type': 'Benders_Optimality', 'count': it_opt_cuts})
        if it_feas_cuts > 0:
            stats_history.append({'time': current_ts, 'type': 'Benders_Feasibility', 'count': it_feas_cuts})
        if math.isfinite(current_ub):
            if current_ub < best_obj - 1e-9:
                best_obj = current_ub
                best_sol_values = sol_values.copy()
                add_objective_cutoff(best_obj)
                if debug: print(f" [NewBest:{best_obj:.4f}]", end="")

            gap = abs(best_obj - lower_bound) / max(1.0, abs(best_obj))
            if debug: print(f", UB={best_obj:.4f}, Gap={gap * 100:.2f}%", end="")
        else:
            if debug: print(", UB=inf", end="")

        if debug: print(f", Cuts={cuts_added}")

        convergence_history.append({
            'iter': it,
            'time': time.time() - solve_start,
            'lb': lower_bound,
            'ub': best_obj if math.isfinite(best_obj) else None
        })

        if math.isfinite(best_obj) and math.isfinite(lower_bound):
            current_gap = abs(best_obj - lower_bound) / max(1e-10, abs(best_obj))

            if current_gap <= 1e-4 :#or (cuts_added == 0 and current_gap <= 1e-3):
                if debug: print(f"Converged (Gap={current_gap:.6f}).")
                break
    if not best_sol_values:
        print("No feasible solution found or timeout.")
        model.end()
        if convergence_history:
            plot_convergence_chart(convergence_history)
        return None

    # --- Final Extraction ---
    rank_to_block = {}
    for k in range(1, res['num_sections'] + 1):
        for b in block_keys:
            if best_sol_values.get(res['r'][k, b], 0.0) > 0.5: rank_to_block[k] = b; break
    drop_order = tuple(rank_to_block.get(k) for k in range(1, res['num_sections'] + 1) if k in rank_to_block)
    best_Perm = tuple(reversed(drop_order)) if drop_order else ()
    lengths_best = [int(round(best_sol_values.get(res['lengths'][k], 0))) for k in range(res['num_sections'])]
    sections_best = []
    start_n = 'SS'
    for k in range(1, res['num_sections'] + 1):
        end_n = rank_to_block.get(k)
        if end_n: sections_best.append((start_n, end_n)); start_n = end_n

    cost_final, risk_final, risk_details, paths = evaluate_final_solution_from_y(
        G=G, y_vars=res['y'], L=res['train_length'], sections=sections_best,
        lengths=lengths_best, sol_values=best_sol_values, alpha=alpha, debug=debug
    )
    solve_time = time.time() - solve_start
    weighted_obj = (alpha * cost_final) + ((1.0 - alpha) * risk_final)

    if math.isfinite(weighted_obj) and math.isfinite(final_lower_bound):
        final_algo_gap = abs(weighted_obj - final_lower_bound) / max(1e-10, abs(weighted_obj))
    else:
        final_algo_gap = 1.0

    model.end()
    plot_convergence_chart(convergence_history)

    try:
        if paths and sections_best:
            full_path_nodes = []

            sorted_indices = sorted(paths.keys())
            for idx in sorted_indices:
                segment_nodes = paths[idx]
                if not segment_nodes: continue

                if full_path_nodes:
                    full_path_nodes.extend(segment_nodes[1:])
                else:
                    full_path_nodes.extend(segment_nodes)

            draw_q1_style(G, full_path_nodes, "LBBD Method: Risk-Aware Routing", cost_final, risk_final, solve_time, is_sequential=False)
    except Exception as e:
        print(f"Q1 Visualization failed: {e}")
    # >>>>> END VISUALIZATION <<<<<
    if stats_history:
        plot_stats_chart(stats_history,convergence_history)

    return {
        'best_Perm': best_Perm,
        'final_order': get_final_order(res['X'], best_sol_values, best_Perm),
        'cost': cost_final,
        'risk': risk_final,
        'weighted_obj': weighted_obj,
        'solve_time': solve_time,
        'paths': paths,
        'risk_details': risk_details,
        'sections': sections_best,
        'lengths': lengths_best,
        'best_bound': final_lower_bound,
        'cplex_gap': final_algo_gap,
        'history': convergence_history
    }

def precompute_bounds(G: nx.DiGraph, key_nodes: List[str], train_length: int) -> Dict:
    for u, v, d in G.edges(data=True):
        P = d.get('P', [])
        if len(P) < train_length:
            P = list(P) + [0.0] * (train_length - len(P))
        d['P'] = P

    max_cost_edge = max((float(d.get('cost', 0.0)) for _, _, d in G.edges(data=True)), default=0.0)
    min_cost_edge = min((float(d.get('cost', 0.0)) for _, _, d in G.edges(data=True)), default=0.0)
    max_exp = max((float(d.get('exposure', 0.0)) for _, _, d in G.edges(data=True)), default=0.0)
    min_exp = min((float(d.get('exposure', 0.0)) for _, _, d in G.edges(data=True)), default=0.0)
    max_P_per_pos = [
        max((float(d.get('P', [0.0] * train_length)[p]) for _, _, d in G.edges(data=True)), default=0.0)
        for p in range(train_length)
    ]
    min_P_per_pos = [
        min((float(d.get('P', [0.0] * train_length)[p]) for _, _, d in G.edges(data=True)), default=0.0)
        for p in range(train_length)
    ]
    max_path_len = max(1, G.number_of_nodes() - 1)

    def full_risk_weight(u, v, d):
        exp = float(d.get('exposure', 0.0))
        P = d.get('P', [0.0] * train_length)
        return exp * sum(float(p) for p in P)

    max_full_risk_edge = max((full_risk_weight(None, None, d) for _, _, d in G.edges(data=True)), default=0.0)

    global_max_cost = max_cost_edge * max_path_len
    global_max_risk = max_full_risk_edge * max_path_len

    max_risk_per_pos = [
        max_path_len * max(
            (float(d.get('exposure', 0.0)) * float(d.get('P', [0.0] * train_length)[p])
             for _, _, d in G.edges(data=True)),
            default=0.0
        )
        for p in range(train_length)
    ]

    new_min_cost_dict: Dict[str, Dict[str, float]] = {}
    for s in key_nodes:
        try:
            new_min_cost_dict[s] = nx.single_source_dijkstra_path_length(
                G, s, weight=lambda u, v, d: float(d.get('cost', 0.0))
            )
        except Exception:
            new_min_cost_dict[s] = {}

    new_min_full_risk_dict: Dict[str, Dict[str, float]] = {}
    for s in key_nodes:
        try:
            new_min_full_risk_dict[s] = nx.single_source_dijkstra_path_length(G, s, weight=full_risk_weight)
        except Exception:
            new_min_full_risk_dict[s] = {}

    min_risk_dict: Dict[int, Dict[str, Dict[str, float]]] = {}
    for pos in range(1, train_length + 1):
        def pos_weight(u, v, d):
            return float(d.get('exposure', 0.0)) * float(d.get('P', [0.0] * train_length)[pos - 1])

        min_risk_dict[pos] = {}
        for s in key_nodes:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, s, weight=pos_weight)
                min_risk_dict[pos][s] = {t: lengths.get(t, float('inf')) for t in key_nodes if t != s}
            except Exception:
                min_risk_dict[pos][s] = {}

    max_totals = [0.0] * train_length
    for _, _, d in G.edges(data=True):
        exp = float(d.get('exposure', 0.0))
        P = d.get('P', [0.0] * train_length)
        for p in range(train_length):
            max_totals[p] += exp * float(P[p])

    all_min_costs = [
        c for s in new_min_cost_dict for t, c in new_min_cost_dict[s].items()
        if c > 0 and c != float('inf')
    ]
    global_min_section_cost = min(all_min_costs) if all_min_costs else 0.0
    global_max_risk_alt = sum(max_risk_per_pos)
    global_max_risk = min(global_max_risk, global_max_risk_alt)

    return {
        'new_min_cost_dict': new_min_cost_dict,
        'new_min_full_risk_dict': new_min_full_risk_dict,
        'global_max_cost': global_max_cost,
        'global_max_risk': global_max_risk,
        'global_min_section_cost': global_min_section_cost,
        'max_risk_per_pos': max_risk_per_pos,
        'max_cost_edge': max_cost_edge,
        'min_cost_edge': min_cost_edge,
        'max_exp': max_exp,
        'min_exp': min_exp,
        'max_P_per_pos': max_P_per_pos,
        'min_P_per_pos': min_P_per_pos,
        'max_full_risk_edge': max_full_risk_edge,
        'max_path_len': max_path_len,
        'min_risk_dict': min_risk_dict,
        'max_totals': max_totals
    }


def plot_convergence_chart(history: List[Dict]):
    """
    Plots the convergence of Lower Bound vs Upper Bound over iterations.
    """
    if not history:
        return

    iterations = [h['iter'] for h in history]
    lbs = [h['lb'] for h in history]
    ubs = [h['ub'] for h in history]

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, lbs, label='Lower Bound (Master)', color='blue', linewidth=2)

    valid_ubs = [(it, u) for it, u in zip(iterations, ubs) if u is not None]
    if valid_ubs:
        u_iters, u_vals = zip(*valid_ubs)
        plt.plot(u_iters, u_vals, label='Upper Bound (Best Feasible)', color='red', linestyle='--', marker='o',
                 markersize=4)

        final_ub = u_vals[-1]
        final_lb = lbs[-1]
        gap = abs(final_ub - final_lb)
        plt.title(f"Convergence Plot\nFinal Gap: {gap:.4f}")
    else:
        plt.title("Convergence Plot (No Feasible Solution Found)")

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def visualize_final_solution(G, sections, paths, key_nodes, cost, risk, time_val):
    """
    Visualizes the final rail network solution using strict layout from G.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # 1. Extract Used Edges/Nodes
    used_edges = set()
    used_nodes = set()

    sorted_sections = sorted(paths.keys())
    for k in sorted_sections:
        path_nodes = paths[k]
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            used_edges.add((u, v))
            used_nodes.add(u)
            used_nodes.add(v)

    # 2. Layout Handling (Critical Fix)
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        print("Warning: 'pos' not found in G. Visualization might look scrambled.")
        # Fallback layout that tries to minimize edge crossing
        pos = nx.kamada_kawai_layout(G)

        # 3. Colors
    node_colors = []
    node_sizes = []
    label_colors = {}

    for n in G.nodes():
        if n == 'SS':
            node_colors.append('#32CD32')  # Green
            node_sizes.append(800)
            label_colors[n] = 'black'
        elif n in key_nodes:
            node_colors.append('#FFD700')  # Yellow
            node_sizes.append(700)
            label_colors[n] = 'black'
        elif n in used_nodes:
            node_colors.append('black')  # Intermediate Used
            node_sizes.append(400)
            label_colors[n] = 'white'
        else:
            node_colors.append('#D3D3D3')  # Unused
            node_sizes.append(150)
            label_colors[n] = 'gray'

    # 4. Edges
    edge_colors = []
    edge_widths = []
    edge_alphas = []

    for u, v in G.edges():
        if (u, v) in used_edges:
            edge_colors.append('red')
            edge_widths.append(2.5)
            edge_alphas.append(1.0)
        else:
            edge_colors.append('#C0C0C0')
            edge_widths.append(0.5)
            edge_alphas.append(1)

    # 5. Draw
    plt.figure(figsize=(20, 10))  # Wider figure to match rail layout
    ax = plt.gca()

    # Draw logic
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=edge_alphas, arrowstyle='-|>',
                           arrowsize=15)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')

    for n, (x, y) in pos.items():
        plt.text(x, y, str(n), fontsize=9, fontweight='bold',
                 color=label_colors.get(n, 'black'), ha='center', va='center')

    # Info Box
    info_text = (f"RESULTS SUMMARY\n----------------\n"
                 f"Total Cost : {cost:.2f}\n"
                 f"Total Risk : {risk:.4f}\n"
                 f"Solve Time : {time_val:.2f}s")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    plt.text(0.01, 0.02, info_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props, fontfamily='monospace')

    plt.title("LBBD Final Route (Corrected Layout)", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_stats_chart(stats_history: List[Dict], convergence_history: List[Dict] = None):
    """
    رسم نمودار با ضخامت متغیر و برچسب‌های Iteration در وسط صفحه
    """
    if not stats_history:
        return

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(stats_history)

    max_time = df['time'].max()
    if convergence_history:
        max_conv_time = max(h['time'] for h in convergence_history)
        max_time = max(max_time, max_conv_time)

    if max_time < 1.0: max_time = 1.0

    # مرتب‌سازی برای ثبات رنگ‌ها
    cut_types = sorted(df['type'].unique())

    plt.figure(figsize=(14, 8))

    # -------------------------------------------------------
    base_linewidth = 6.0
    colors = ['#2980b9', '#e67e22', '#27ae60', '#8e44ad', '#f1c40f']  # آبی، نارنجی، سبز، بنفش، زرد

    for i, c_type in enumerate(cut_types):
        subset = df[df['type'] == c_type].copy()

        # نقطه شروع (0,0)
        start_row = pd.DataFrame({'time': [0.0], 'count': [0], 'type': [c_type]})
        subset = pd.concat([start_row, subset], ignore_index=True)

        subset = subset.sort_values('time')
        subset['cum_count'] = subset['count'].cumsum()

        # نقطه پایان
        last_cum = subset['cum_count'].iloc[-1]
        end_row = pd.DataFrame({'time': [max_time], 'cum_count': [last_cum], 'type': [c_type]})

        final_plot_data = pd.concat([subset, end_row], ignore_index=True)

        current_lw = max(1.5, base_linewidth - (i * 1.2))
        current_color = colors[i % len(colors)]

        plt.step(final_plot_data['time'], final_plot_data['cum_count'],
                 label=f'{c_type} ({int(last_cum)})',
                 where='post',
                 linewidth=current_lw,
                 color=current_color,
                 alpha=0.95)

    # -------------------------------------------------------
    total_df = df.sort_values('time').copy()
    total_start = pd.DataFrame({'time': [0.0], 'count': [0]})
    total_df = pd.concat([total_start, total_df], ignore_index=True)
    total_df['total_cum'] = total_df['count'].cumsum()

    total_last = total_df['total_cum'].iloc[-1]
    total_end = pd.DataFrame({'time': [max_time], 'total_cum': [total_last]})
    total_plot_data = pd.concat([total_df, total_end], ignore_index=True)

    plt.step(total_plot_data['time'], total_plot_data['total_cum'], label=f'TOTAL ({int(total_last)})',
             color='#2c3e50', linestyle='--', linewidth=3, where='post', alpha=0.8)

    # -------------------------------------------------------
    if convergence_history:
        y_mid = total_last / 2

        for item in convergence_history:
            it_time = item['time']
            it_num = item['iter']

            plt.axvline(x=it_time, color='#c0392b', linestyle='--', alpha=0.6, linewidth=1.5)

            if len(convergence_history) < 40 or it_num % 5 == 0 or it_num == 1:
                plt.text(it_time, y_mid, f'Iter {it_num}',
                         rotation=90,
                         verticalalignment='center',
                         horizontalalignment='center',
                         fontsize=11,
                         fontweight='bold',
                         color='#c0392b',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    plt.title("Dynamic Cut Generation Analysis", fontsize=16)
    plt.xlabel("Elapsed Time (seconds)", fontsize=12)
    plt.ylabel("Cumulative Number of Cuts", fontsize=12)
    plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.tight_layout()
    #plt.savefig("Cuts_Dynamic_Stats.png")
    plt.show()