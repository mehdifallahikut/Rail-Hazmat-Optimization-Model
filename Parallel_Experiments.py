from joblib import Parallel, delayed
import pandas as pd
import time
import random
import gc
import json
import glob
from colorama import Fore
import os

# --- Imports from your modules ---
from data import generate_blocks, generate_rail_network
from graph_utils import to_bidirected
from solution_utils import (
    precompute_min_risk_and_max_totals,
    precompute_global_max
)
from OptPerm_LBBD_Weighted import run_optPerm_LBBD_Weighted

# --- System Configs ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

SAVE_CSV_NAME = "results_paper_C_OR.csv"
TEMP_DIR = "temp_results"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def save_single_result(result_dict, prefix):

    try:
        filename = f"{prefix}_{result_dict.get('Experiment_ID', 'Unk')}_{result_dict.get('Seed', 0)}_{result_dict.get('Model', 'Unk')}.json"
        filepath = os.path.join(TEMP_DIR, filename)

        with open(filepath, 'w') as f:
            json.dump(result_dict, f)

    except Exception as e:
        print(f"{Fore.RED}Critical Error Saving Temp File: {e}{Fore.RESET}")


# --- Main Experiment Task ---
def run_experiment_row(config, seed):
    random.seed(seed)

    try:
        blocks = generate_blocks(config['n_blocks'], config['haz_pct'], 0.5, config['len'])
        block_keys = list(blocks.keys())
        total_nodes = 10 * len(blocks)
        network_graph = generate_rail_network(total_nodes, block_keys, 0.5, config['len'])
        G = to_bidirected(network_graph)
        key_nodes = ['SS'] + block_keys

        min_risk_dict, max_totals = precompute_min_risk_and_max_totals(G, key_nodes, config['len'])
        global_max_cost, global_max_risk = precompute_global_max(G, blocks, config['len'])
    except Exception as e:
        err_row = {'Status': f"DataGen Error: {e}", 'Experiment_ID': config['id'], 'Seed': seed, 'Model': 'DataGen'}
        save_single_result(err_row, "MAIN")
        return [err_row]

    results_list = []

    for model_name in config['models']:
        expected_filename = f"MAIN_{config['id']}_{seed}_{model_name}.json"
        if os.path.exists(os.path.join(TEMP_DIR, expected_filename)):
            print(f"Skipping {expected_filename} (Already exists)")
            continue

        start_time = time.time()
        res = None
        status = "Solved"

        try:
            if model_name == 'M1':
                res = run_model_opt(block_keys, blocks, config['len'], G,
                                    min_risk_dict, max_totals,
                                    global_max_cost, global_max_risk,
                                    timeout=config['timeout'])
                print(config, seed, model_name)
            elif model_name == 'M2':
                res = run_model_opt_vi(block_keys, blocks, config['len'], G,
                                       min_risk_dict, max_totals,
                                       global_max_cost, global_max_risk,
                                       timeout=config['timeout'])
                print(config, seed, model_name)
            elif model_name == 'M3':
                res = run_optPerm_LBBD_Weighted(block_keys, blocks, config['len'], G,
                                                timeout=config['timeout'],
                                                alpha=0.5)
                print(config, seed, model_name)
        except Exception as e:
            status = f"Error: {e}"

        duration = time.time() - start_time

        final_status = status
        if res and duration > config['timeout']:
            final_status = "TimeLimit"
        elif res is None and status == "Solved":
            final_status = "TimeOut/Null"

        row = {
            'Experiment_ID': config['id'],
            'Seed': seed,
            'Model': model_name,
            'Time': duration,
            'Blocks': config['n_blocks'],
            'Length': config['len'],
            'Hazmat_Pct': config['haz_pct'],
            'Cost': res['cost'] if res else None,
            'Risk': res['risk'] if res else None,
            'Gap': 0.0,
            'Status': final_status,
            'Best_Bound': res.get('best_bound', 0.0) if res else 0.0,
            'CPLEX_Gap': res.get('cplex_gap', 1.0) if res else 1.0
        }

        save_single_result(row, "MAIN")
        results_list.append(row)

    del G, network_graph, min_risk_dict
    gc.collect()

    return results_list


# --- Pareto Task ---
def run_pareto_row(alpha, seed, timeout):
    random.seed(seed)

    expected_filename = f"PARETO_Pareto_{seed}_M3_Alpha_{alpha}.json"
    if os.path.exists(os.path.join(TEMP_DIR, expected_filename)):
        print(f"Skipping {expected_filename} (Already exists)")
        return None

    try:
        p_blocks = generate_blocks(5, 0.5, 0.5, 150)
        p_keys = list(p_blocks.keys())
        total_nodes = 10 * 5
        p_graph = generate_rail_network(total_nodes, p_keys, 0.5, 150)
        G_p = to_bidirected(p_graph)
    except Exception as e:
        err_row = {'Status': f"DataGen Error: {e}", 'Experiment_ID': 'Pareto', 'Model': f'M3_Alpha_{alpha}'}
        save_single_result(err_row, "PARETO")
        return err_row

    st = time.time()
    try:
        res_p = run_optPerm_LBBD_Weighted(p_keys, p_blocks, 150, G_p, timeout=timeout, alpha=alpha)
        print(seed, alpha)
        status = 'Solved' if res_p else 'Fail'
    except Exception as e:
        res_p = None
        status = f"Error: {e}"

    dur = time.time() - st

    row = {
        'Experiment_ID': 'Pareto',
        'Seed': seed,
        'Model': f'M3_Alpha_{alpha}',
        'Time': dur,
        'Blocks': 5, 'Length': 150, 'Hazmat_Pct': 0.5,
        'Cost': res_p['cost'] if res_p else None,
        'Risk': res_p['risk'] if res_p else None,
        'Gap': 0.0,
        'Status': status,
        'Best_Bound': res_p.get('best_bound', 0.0) if res_p else 0.0,
        'CPLEX_Gap': res_p.get('cplex_gap', 1.0) if res_p else 1.0
    }

    # === ذخیره لحظه‌ای ===
    save_single_result(row, "PARETO")

    del G_p, p_blocks
    gc.collect()

    return row


def combine_results_to_csv():
    """
    تجمیع تمام فایل‌های JSON موقت در یک فایل CSV نهایی
    """
    print(f"\n{Fore.YELLOW}Combining temp files into final CSV...{Fore.RESET}")
    all_files = glob.glob(os.path.join(TEMP_DIR, "*.json"))
    combined_data = []
    for f in all_files:
        try:
            with open(f, 'r') as infile:
                data = json.load(infile)
                if data:  # مطمئن شویم فایل خالی نیست
                    combined_data.append(data)
        except:
            pass

    if combined_data:
        df = pd.DataFrame(combined_data)

        # مرتب‌سازی ستون‌ها
        cols_order = ['Experiment_ID', 'Seed', 'Model', 'Time', 'Cost', 'Risk', 'Status', 'CPLEX_Gap', 'Best_Bound']
        existing_cols = [c for c in cols_order if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in cols_order]
        df = df[existing_cols + remaining_cols]

        if 'Experiment_ID' in df.columns and 'Seed' in df.columns:
            df = df.sort_values(by=['Experiment_ID', 'Seed'])

        df.to_csv(SAVE_CSV_NAME, index=False)
        print(f"{Fore.GREEN}Successfully saved {len(df)} rows to {SAVE_CSV_NAME}{Fore.RESET}")
    else:
        print(f"{Fore.RED}No results found in {TEMP_DIR}{Fore.RESET}")


# --- تابع اصلی ---
def run_paper_experiments():
    print(f"{Fore.CYAN}=== Starting PARALLEL Experiments for Computers & OR (SAFE MODE) ==={Fore.RESET}")

    # 1. تعریف سناریوها
    experiment_configs = [
        # --- Small Instances ---
        # {'id': 'Small1', 'n_blocks': 3, 'len': 100, 'haz_pct': 0.5, 'timeout': 3600, 'models': ['M1', 'M2', 'M3']},
        # {'id': 'Small2', 'n_blocks': 3, 'len': 150, 'haz_pct': 0.5, 'timeout': 3600, 'models': ['M1', 'M2', 'M3']},
        # {'id': 'Small3', 'n_blocks': 3, 'len': 250, 'haz_pct': 0.5, 'timeout': 3600, 'models': ['M1', 'M2', 'M3']},

        # --- Medium Instances ---
        # {'id': 'Medium1', 'n_blocks': 5, 'len': 100, 'haz_pct': 0.5, 'timeout': 3600, 'models': ['M1', 'M2', 'M3']},
        #{'id': 'Medium2', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.5, 'timeout': 3600, 'models': ['M1', 'M2', 'M3']},
        # {'id': 'Medium3', 'n_blocks': 5, 'len': 250, 'haz_pct': 0.5, 'timeout': 3600, 'models': ['M1', 'M2', 'M3']},

        # --- Large Instances ---
        # {'id': 'Large1', 'n_blocks': 7, 'len': 100, 'haz_pct': 0.5, 'timeout': 3600, 'models': [ 'M3']},
        # {'id': 'Large2', 'n_blocks': 7, 'len': 150, 'haz_pct': 0.5, 'timeout': 3600, 'models': [ 'M3']},
        # {'id': 'Large3', 'n_blocks': 7, 'len': 250, 'haz_pct': 0.5, 'timeout': 3600, 'models': [ 'M3']},

        # --- Sensitivity ---
        {'id': 'HazSen1', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.1, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen1.5', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.15, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen2', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.2, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen2.5', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.25, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen3', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.3, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen3.5', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.35, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen4', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.4, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen4.5', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.45, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen5', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.5, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen5.5', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.55, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen6', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.6, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen6.5', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.65, 'timeout': 3600, 'models': ['M3']},
        {'id': 'HazSen7', 'n_blocks': 5, 'len': 150, 'haz_pct': 0.7, 'timeout': 3600, 'models': ['M3']},
    ]

    seeds = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

    # 2. آماده‌سازی تسک‌ها
    tasks = []
    for config in experiment_configs:
        for seed in seeds:
            tasks.append((config, seed))

    # --- تنظیمات ایمنی اجرا ---
    SAFE_N_JOBS = 6

    print(f"{Fore.YELLOW}>> Running {len(tasks)} main experiments in PARALLEL (n_jobs={SAFE_N_JOBS})...{Fore.RESET}")

    # اجرای موازی
    Parallel(n_jobs=SAFE_N_JOBS, backend='loky', verbose=5)(
        delayed(run_experiment_row)(c, s) for c, s in tasks
    )

    # 3. بخش پارتو
    print(f"\n{Fore.MAGENTA}>> Running Pareto Analysis in PARALLEL (M3 Only){Fore.RESET}")
    pareto_seed = 43
    alpha_steps = [0.0,0.05, 0.1,0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95, 1.0]
    alpha_steps=[0.5]
    pareto_timeout = 3600

    Parallel(n_jobs=SAFE_N_JOBS, backend='loky', verbose=5)(
        delayed(run_pareto_row)(alpha, pareto_seed, pareto_timeout) for alpha in alpha_steps
    )
    # 4. ذخیره نهایی
    combine_results_to_csv()
    print(f"\n{Fore.GREEN}All experiments completed.{Fore.RESET}")

if __name__ == "__main__":
    run_paper_experiments()