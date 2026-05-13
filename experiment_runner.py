"""
experiment_runner.py — fair 60,000 eval budget across all algorithms.

HOA/Hybrid have 3 phases per iteration = 3×pop evals per iter.
Single-phase algos (CS, DO, EO, RIME, FOX, GTO, AO, AVOA) = 1×pop per iter.

Fair Budgeting:
    HOA / Hybrid:  pop=30, max_iter=666   → 59,940 evals
    All others:    pop=30, max_iter=2000  → 60,000 evals
"""

import numpy as np
import csv, os, time
from scipy.stats import wilcoxon

from algorithms.hoa_base import HOA
from algorithms.cs_base import CuckooSearch
from algorithms.hybrid import HybridHOACS
from algorithms.comparison_algorithms import COMPARISON_ALGORITHMS

# CONFIG

BENCHMARK_YEAR = 2014  # Toggle: 2014 or 2017
N_RUNS   = 15
DIM      = 10
LB, UB   = -100, 100
CEC_SEED = 42

# Select representative functions
if BENCHMARK_YEAR == 2014:
    from benchmarks.cec2014 import get_function, FUNCTION_INFO
    FUNC_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
else:
    from benchmarks.cec2017 import get_function, FUNCTION_INFO
    FUNC_IDS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

OUTPUT_DIR = "results"

# Evaluation parameters
HOA_POP,    HOA_ITER    = 30, 666
HYBRID_POP, HYBRID_ITER = 30, 666
OTHER_POP,  OTHER_ITER  = 30, 2000

# Algorithm Registry

def make_solver(name, cls, obj_func):
    if name in ["HOA", "Hybrid"]:
        return cls(obj_func, DIM, HOA_POP, HOA_ITER, LB, UB)
    return cls(obj_func, DIM, OTHER_POP, OTHER_ITER, LB, UB)

ALL_ALGORITHMS = {
    "HOA"    : HOA,
    "CS"     : CuckooSearch,
    "Hybrid" : HybridHOACS,
    **COMPARISON_ALGORITHMS,
}
ALGO_NAMES = list(ALL_ALGORITHMS.keys())

# Core Runner

def run_all(verbose=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_algos, n_funcs = len(ALGO_NAMES), len(FUNC_IDS)
    errors = np.full((n_algos, n_funcs, N_RUNS), np.nan)

    for fi, fid in enumerate(FUNC_IDS):
        func = get_function(fid, dim=DIM, shift=True, rotate=True, seed=CEC_SEED)
        _, fname, fcat = FUNCTION_INFO[fid]

        if verbose:
            print(f"\n[F{fid:02d} - CEC{BENCHMARK_YEAR}] {fname} ({fcat})")
            print(f"  {'Algorithm':<10}  {'Mean Error':>14}  {'Std':>12}  {'Time':>8}")
            print(f"  {'-'*52}")

        for ai, algo_name in enumerate(ALGO_NAMES):
            cls = ALL_ALGORITHMS[algo_name]
            run_errors = []
            t0 = time.time()

            for run in range(N_RUNS):
                np.random.seed(run * 1000 + fid) # Reproducibility
                solver = make_solver(algo_name, cls, func)
                _, best_fit, _ = solver.solve()
                run_errors.append(max(best_fit - func.bias, 0.0))

            errors[ai, fi, :] = run_errors
            elapsed = time.time() - t0

            if verbose:
                print(f"  {algo_name:<10}  {np.mean(run_errors):>14.4e}  "
                      f"{np.std(run_errors):>12.4e}  {elapsed:>7.1f}s")

    np.save(os.path.join(OUTPUT_DIR, f"raw_errors_cec{BENCHMARK_YEAR}.npy"), errors)
    return errors

# Statistical Analysis (Wilcoxon)

def build_wilcoxon_table(errors):
    hi = ALGO_NAMES.index("Hybrid")
    wt = {}
    for fi, fid in enumerate(FUNC_IDS):
        wt[fid] = {}
        h_err = errors[hi, fi, :]
        for ai, name in enumerate(ALGO_NAMES):
            if name == "Hybrid":
                wt[fid][name] = (None, 1.0, "—")
                continue
            
            target_err = errors[ai, fi, :]
            
            if np.array_equal(h_err, target_err):
                wt[fid][name] = (0.0, 1.0, "=")
                continue
                
            try:
                stat, p = wilcoxon(h_err, target_err)
                # +: Hybrid is significantly better (lower mean error)
                if p < 0.05 and np.mean(h_err) < np.mean(target_err):
                    sym = "+"
                # -: Hybrid is significantly worse
                elif p < 0.05 and np.mean(h_err) > np.mean(target_err):
                    sym = "-"
                # =: No significant difference
                else:
                    sym = "="
            except ValueError:
                stat, p, sym = 0.0, 1.0, "="
            wt[fid][name] = (stat, p, sym)
    return wt

# Export Report

def export_txt_report(errors, wt):
    path = os.path.join(OUTPUT_DIR, f"final_report_cec{BENCHMARK_YEAR}.txt")
    n_algos = len(ALGO_NAMES)
    
    # Calculate Ranks based on mean error per function
    means = np.mean(errors, axis=2) 
    ranks = np.zeros_like(means)
    for f in range(len(FUNC_IDS)):
        order = means[:, f].argsort()
        ranks[order, f] = np.arange(1, n_algos + 1)
    avg_ranks = np.mean(ranks, axis=1)

    with open(path, "w") as f:
        f.write(f"EXPERIMENT REPORT: CEC{BENCHMARK_YEAR}\n")
        f.write(f"Setup: DIM={DIM}, Runs={N_RUNS}, Budget=60,000 FEs\n")
        f.write("-" * 65 + "\n")
        
        header = f"{'Algorithm':<15} | {'Avg Rank':<10} | {'W/T/L vs Hybrid'}"
        f.write(header + "\n" + "-" * len(header) + "\n")

        for ai, name in enumerate(ALGO_NAMES):
            if name == "Hybrid":
                wt_str = "Reference"
            else:
                wins = sum(1 for fid in FUNC_IDS if wt[fid][name][2] == "+")
                ties = sum(1 for fid in FUNC_IDS if wt[fid][name][2] == "=")
                loss = sum(1 for fid in FUNC_IDS if wt[fid][name][2] == "-")
                wt_str = f"{wins}/{ties}/{loss}"
            
            f.write(f"{name:<15} | {avg_ranks[ai]:<10.2f} | {wt_str}\n")

        f.write("\nNote: W/T/L corresponds to (Better/Tie/Worse) compared to Hybrid.\n")
        f.write("Significance level: p < 0.05\n")

    print(f"\nExperiment report generated: {path}")

if __name__ == "__main__":
    print(f"Starting fair-budget benchmark for CEC{BENCHMARK_YEAR}...")
    errors = run_all()
    wt = build_wilcoxon_table(errors)
    export_txt_report(errors, wt)