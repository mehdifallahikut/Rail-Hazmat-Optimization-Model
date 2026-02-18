# new_main.py
import gc
import itertools
import random
import time
import networkx as nx
from colorama import Fore
import matplotlib.pyplot as plt
from OptPerm_LBBD_Weighted import run_optPerm_LBBD_Weighted
from data import generate_blocks, generate_rail_network, LAMBDA_MATRIX
from Parallel_Experiments import run_paper_experiments
from graph_utils import draw_network, to_bidirected
from multi_astar import calculate_total_shunting_operations
from solution_utils import (
    precompute_min_risk_and_max_totals,
    precompute_global_max,
    validate_cost_and_risk,
    print_path_cost_risk_details,
    validate_safety_constraints
)

# ==========================================
#  Configuration & Flags
# ==========================================
RUN_EXPERIMENT_MODE = False  # Set to True for paper experiments
# Model Flags
optPerm_flag = False  # Standard MIP
optPerm_VI_flag = False  # MIP with VI
optPerm_LBBD_Weighted_flag = True  # New: Weighted Objective
run_benchmarks_flag = False  # Run CostOnly (V0) and RiskOnly (V1) benchmarks

# Settings
print_Flag = False
VALIDATION_ENABLED = True  # Enable detailed validation checks

# ==========================================
#  Main Execution
# ==========================================
def main():
    if RUN_EXPERIMENT_MODE:
        run_paper_experiments()
        return

    all_time = time.time()
    seed=43
    random.seed(seed)

    # Parameters
    hazmat_percentage = 0.5
    hazmat_diversity = 0.5
    train_length =100
    number_of_blocks = 3
    timeout = 36000
    weighted_alpha = 0.5 # For Weighted LBBD (0.5 = Balanced) 0 risk 1 cost

    # Data Generation
    blocks = generate_blocks(number_of_blocks, hazmat_percentage, hazmat_diversity, train_length)
    block_keys = list(blocks.keys())
    all_permutations = list(sorted(itertools.permutations(block_keys), reverse=True))

    total_nodes = 10 * len(blocks)
    variance_level = 0.5
    network_graph = generate_rail_network(total_nodes, block_keys, variance_level, train_length)

    # Visualization (Optional)
    plt.figure(figsize=(25, 10))
    pos = nx.get_node_attributes(network_graph, 'pos')
    key_nodes = ['SS'] + block_keys
    draw_network(network_graph, key_nodes)

    G: nx.DiGraph = to_bidirected(network_graph)
    del network_graph  # Free memory

    # Precomputations
    min_risk_dict, max_totals = precompute_min_risk_and_max_totals(G, ['SS'] + block_keys, train_length)
    global_max_cost, global_max_risk = precompute_global_max(G, blocks, train_length)

    # Header Printing
    print("=====================================================================================================================================================")
    print(f"{'Permutation':<30}\tBlock\tTrL\tCost\tRisk\t\tTime\t\tNumber of Shuting Operations\t\tSh_Time\t\tModel\t\t\t\t\tSeed")
    print("=====================================================================================================================================================")

    # ---------------------------------------------------------
    #  OptPerm_LBBD_Weighted ( Weighted Objective)
    # ---------------------------------------------------------
    if optPerm_LBBD_Weighted_flag:
        # Run with alpha = 0.5 (or whatever variable you set)
        result_lbbd_weighted = run_optPerm_LBBD_Weighted(
            block_keys, blocks, train_length, G,
            timeout=timeout,
            alpha=weighted_alpha,
            debug=False
        )
        process_model_result(result_lbbd_weighted, f"LBBD-Weighted(a={weighted_alpha})", blocks, train_length, G,
                             print_Flag,seed)
    # ---------------------------------------------------------


    # Final Summary
    print("=======================================================================================================================")
    print("Number of Blocks:",number_of_blocks, "\tTrain Lenght:",train_length)
    print(f"Total Experiment Time: {(time.time() - all_time):.1f} sec")
    gc.collect()


# ==========================================
#  Helper Function to Standardize Output
# ==========================================
def process_model_result(result, model_name, blocks, train_length, G, print_flag=False,seed=0):
    """
    Standardizes output printing, operations calculation, and validation for all models.
    """
    if result is None:
        print(f"{Fore.RED}No solution found for {model_name}{Fore.RESET}")
        return None

    # 1. Normalize Result Keys (Different models might use different keys)
    perm = result.get('perm') or result.get('best_Perm')
    final_order = result.get('final_order')
    cost = result.get('cost')
    risk = result.get('risk')
    solve_time = result.get('solve_time')

    # 2. Print Basic Stats (Permutation, Cost, Risk, Time)
    # Using specific formatting to match your request
    print(f"{str(perm):<30}\t", end="")
    print(f"{Fore.YELLOW}{len(blocks):.0f}{Fore.RESET}\t\t", end="")
    print(f"{Fore.YELLOW}{train_length:.0f}{Fore.RESET}\t", end="")
    print(f"{Fore.LIGHTYELLOW_EX}{cost:.1f}{Fore.RESET}\t", end="")
    print(f"{Fore.LIGHTRED_EX}{risk:.4f}{Fore.RESET}\t", end="")
    print(f"{Fore.LIGHTCYAN_EX}{solve_time:.4f}{Fore.RESET}\t\t", end="")


    # 3. Calculate Shunting Operations
    calc_start = time.time()
    ops, callback, _ = calculate_total_shunting_operations(perm, blocks, final_order, print_flag)
    ops_time = time.time() - calc_start

    ops_color = Fore.LIGHTGREEN_EX if callback else Fore.LIGHTGREEN_EX  # Keep green but maybe indicate fallback in text
    fallback_msg = f"\t{Fore.LIGHTRED_EX}Fall back Needed{Fore.RESET}" if not callback else ""

    print(f"{ops_color}{ops}{Fore.RESET}\t\t\t\t\t", end="\t\t\t\t")
    print(f"{Fore.LIGHTMAGENTA_EX}{ops_time:.4f}{Fore.RESET}\t", end="\t")
    print(f"{Fore.CYAN}{model_name}{Fore.RESET}{fallback_msg}\t", end="")
    print(f"{Fore.YELLOW}{seed:.1f}{Fore.RESET}")

    # 4. Validations
    if VALIDATION_ENABLED:
        # A. Cost/Risk Re-calculation check
        if 'sections' in result and 'lengths' in result and 'paths' in result:
            val_res = validate_cost_and_risk(
                perm, final_order,
                result['sections'], result['lengths'], result['paths'],
                train_length, G
            )

        # B. Safety Constraints Check
        isValid, errors = validate_safety_constraints(perm, final_order)
        if not isValid:
            print(f"   {Fore.RED}!!! SAFETY VIOLATION in {model_name} !!!{Fore.RESET}")
            for err in errors:
                print(f"   - {err}")
            print(f"   Order: {final_order}")

    return ops

if __name__ == "__main__":
    main()