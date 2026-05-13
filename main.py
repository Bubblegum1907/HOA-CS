import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from algorithms.hoa_base import HOA
from algorithms.cs_base import cuckoo_search, CuckooSearch
from algorithms.hybrid import HybridHOACS
from benchmarks.cec2014 import get_function, FUNCTION_INFO
from benchmarks.cec2017 import get_function, FUNCTION_INFO
from algorithms.comparison_algorithms import COMPARISON_ALGORITHMS
from utils.visualisation import plot_convergence, plot_accuracy_comparison

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# CONFIG
# =============================================================================

CEC_DIM       = 10
CEC_POP_SIZE  = 50
CEC_MAX_ITER  = 400
CEC_LB, CEC_UB = -100, 100
CEC_SEED      = 42

CEC_FUNC_IDS  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
CEC2017_FUNC_IDS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


ALGORITHMS = {
    "HOA"    : lambda obj, dim, pop, itr, lb, ub: HOA(obj, dim, pop, itr, lb, ub),
    "CS"     : lambda obj, dim, pop, itr, lb, ub: CuckooSearch(obj, dim, pop, itr, lb, ub),
    "Hybrid" : lambda obj, dim, pop, itr, lb, ub: HybridHOACS(obj, dim, pop, itr, lb, ub),
    **{name: (lambda cls: lambda obj, dim, pop, itr, lb, ub: cls(obj, dim, pop, itr, lb, ub))(cls)
       for name, cls in COMPARISON_ALGORITHMS.items()}
}

COLORS = {"HOA": "#7f8c8d", "CS": "#2980b9", "Hybrid": "#e74c3c"}
LINES  = {"HOA": "--",      "CS": "-.",       "Hybrid": "-"}


# SECTION 1 — CEC2014 Benchmark

def run_cec_benchmark():
    print("\n" + "=" * 60)
    print("  CEC2014 BENCHMARK")
    print("=" * 60)

    # results[func_id][algo_name] = (best_fit, curve, error)
    results = {fid: {} for fid in CEC_FUNC_IDS}

    for fid in CEC_FUNC_IDS:
        func = get_function(fid, dim=CEC_DIM, shift=True, rotate=True, seed=CEC_SEED)
        label, name, category = FUNCTION_INFO[fid]
        print(f"\n  F{fid:02d} — {name} [{category}]  (bias={func.bias})")
        print(f"  {'Algorithm':<10}  {'Best Fit':>14}  {'Error':>14}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}")

        for algo_name, factory in ALGORITHMS.items():
            solver = factory(func, CEC_DIM, CEC_POP_SIZE, CEC_MAX_ITER, CEC_LB, CEC_UB)
            _, best_fit, curve = solver.solve()
            error = best_fit - func.bias
            results[fid][algo_name] = (best_fit, curve, error)
            print(f"  {algo_name:<10}  {best_fit:>14.4f}  {error:>14.4f}")

    return results


def plot_cec_convergence(results):
    """One convergence subplot per tested CEC function."""
    n = len(CEC_FUNC_IDS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes = axes.flatten()

    for ax_idx, fid in enumerate(CEC_FUNC_IDS):
        ax = axes[ax_idx]
        _, name, category = FUNCTION_INFO[fid]

        for algo_name in ALGORITHMS:
            _, curve, _ = results[fid][algo_name]
            ax.plot(curve,
                    label=algo_name,
                    color=COLORS[algo_name],
                    linestyle=LINES[algo_name],
                    linewidth=1.8 if algo_name == "Hybrid" else 1.3)

        ax.set_title(f"F{fid} — {name[:28]}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=7)
        ax.set_ylabel("Fitness", fontsize=7)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")

    for ax in axes[len(CEC_FUNC_IDS):]:
        ax.set_visible(False)

    fig.suptitle(f"CEC2014 Convergence Comparison  (dim={CEC_DIM})",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("cec2014_convergence.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("\n  Saved → cec2014_convergence.png")


def plot_cec_error_bars(results):
    """Grouped bar chart of log-error per function for each algorithm."""
    fids   = CEC_FUNC_IDS
    algo_names = list(ALGORITHMS.keys())
    n_funcs = len(fids)
    n_algos = len(algo_names)
    x = np.arange(n_funcs)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n_funcs * 0.9), 5))

    for i, algo_name in enumerate(algo_names):
        errors = [max(results[fid][algo_name][2], 1e-10) for fid in fids]
        bars = ax.bar(x + i * width, errors, width,
                      label=algo_name,
                      color=COLORS[algo_name],
                      alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"F{fid}" for fid in fids], fontsize=9)
    ax.set_ylabel("Error (log scale, lower is better)", fontsize=10)
    ax.set_title(f"CEC2014 Error Comparison  (dim={CEC_DIM}, iter={CEC_MAX_ITER})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("cec2014_error_comparison.png", dpi=200)
    plt.show()
    print("  Saved → cec2014_error_comparison.png")

def run_cec2017_benchmark():
    from benchmarks.cec2017 import get_function as get_f17, FUNCTION_INFO as INFO_17
    
    print("\n" + "=" * 60)
    print("  CEC2017 BENCHMARK")
    print("=" * 60)

    results = {fid: {} for fid in CEC2017_FUNC_IDS}

    for fid in CEC2017_FUNC_IDS:
        func = get_f17(fid, dim=CEC_DIM, shift=True, rotate=True, seed=CEC_SEED)
        label, name, category = INFO_17[fid]
        
        print(f"\n  F{fid:02d} — {name} [{category}]")
        print(f"  {'Algorithm':<10}  {'Best Fit':>14}  {'Error':>14}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}")

        for algo_name, factory in ALGORITHMS.items():
            solver = factory(func, CEC_DIM, CEC_POP_SIZE, CEC_MAX_ITER, CEC_LB, CEC_UB)
            _, best_fit, curve = solver.solve()
            
            error = best_fit - func.bias
            results[fid][algo_name] = (best_fit, curve, error)
            print(f"  {algo_name:<10}  {best_fit:>14.4f}  {error:>14.4e}")

    return results

def plot_cec_generic_convergence(results, func_ids, info_dict, title, filename):
    """Generalized plotting for both 2014 and 2017."""
    n = len(func_ids)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    axes = axes.flatten()

    for ax_idx, fid in enumerate(func_ids):
        ax = axes[ax_idx]
        _, name, _ = info_dict[fid]

        for algo_name in ALGORITHMS:
            _, curve, _ = results[fid][algo_name]
            ax.plot(curve, label=algo_name, color=COLORS[algo_name], 
                    linestyle=LINES[algo_name], alpha=0.9)

        ax.set_title(f"F{fid}: {name[:20]}", fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        if ax_idx == 0: ax.legend(fontsize=6)

    for ax in axes[len(func_ids):]: ax.set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


if __name__ == "__main__":
    from benchmarks.cec2014 import FUNCTION_INFO as INFO_14
    from benchmarks.cec2017 import FUNCTION_INFO as INFO_17

    results_14 = run_cec_benchmark()
    plot_cec_generic_convergence(results_14, CEC_FUNC_IDS, INFO_14, 
                                 "CEC2014 Convergence Comparison", "cec2014_convergence.png")
    plot_cec_error_bars(results_14) 

    results_17 = run_cec2017_benchmark()
    plot_cec_generic_convergence(results_17, CEC2017_FUNC_IDS, INFO_17, 
                                 "CEC2017 Convergence Comparison", "cec2017_convergence.png")

    
    print("\nAll benchmarks completed and plots saved.")