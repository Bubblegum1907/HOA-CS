import os
import sys
import argparse
import numpy as np
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from algorithms.hoa_base import HOA
from algorithms.cs_base  import cuckoo_search
from algorithms.hybrid   import HybridHOACS
from benchmarks.engineering_problems import get_problem, ALL_PROBLEMS

POP_SIZE      = 30
N_RUNS        = 50
MAX_FES       = 60_000
MAX_ITER      = MAX_FES // POP_SIZE   # 2000
N_CHECKPOINTS = 500
SAVE_DIR      = 'results'            
FIG_DIR       = 'figures_engineering'
ALPHA         = 0.05
ALGO_NAMES    = ['HOA', 'CS', 'HybridHOACS']

STYLES = {
    'HybridHOACS': {'color': '#D85A30', 'lw': 2.2, 'dash': []},
    'HOA':         {'color': '#1D9E75', 'lw': 1.4, 'dash': [6, 3]},
    'CS':          {'color': '#7F77DD', 'lw': 1.4, 'dash': [4, 4]},
}


class EngineeringWrapper:
    def __init__(self, problem_name: str):
        self._prob      = get_problem(problem_name)
        self.name       = self._prob.name
        self.dim        = self._prob.dim
        self.lb         = self._prob.lb
        self.ub         = self._prob.ub
        self.best_known = self._prob.best_known

    def __call__(self, x: np.ndarray) -> float:
        return float(self._prob(x))

    def violation(self, x)   -> float: return self._prob.violation(x)
    def is_feasible(self, x) -> bool:  return self._prob.is_feasible(x)


def run_full_experiment(problem_names=ALL_PROBLEMS, pop_size=POP_SIZE,
                        max_iter=MAX_ITER, n_runs=N_RUNS,
                        n_checkpoints=N_CHECKPOINTS, save_dir=SAVE_DIR) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    results = {a: {} for a in ALGO_NAMES}

    for pname in problem_names:
        wrap = EngineeringWrapper(pname)
        print(f'\n{"═"*62}')
        print(f'  {wrap.name}  |  dim={wrap.dim}  |  known≈{wrap.best_known}')
        print(f'  {n_runs} runs × {pop_size}pop × {max_iter}iter = {pop_size*max_iter:,} FEs')
        print(f'{"═"*62}')

        all_fits   = {a: np.zeros(n_runs)                  for a in ALGO_NAMES}
        all_sols   = {a: np.zeros((n_runs, wrap.dim))      for a in ALGO_NAMES}
        all_feas   = {a: np.zeros(n_runs, dtype=bool)      for a in ALGO_NAMES}
        all_curves = {a: np.zeros((n_runs, n_checkpoints)) for a in ALGO_NAMES}

        for run in range(n_runs):
            np.random.seed(run * 77 + hash(pname) % 997)
            print(f'  run {run+1:02d}/{n_runs}  ', end='', flush=True)

            # HOA
            hoa = HOA(wrap, wrap.dim, pop_size, max_iter, wrap.lb, wrap.ub)
            h_sol, h_fit, h_curve = hoa.solve()
            _record(all_fits, all_sols, all_feas, all_curves,
                    'HOA', run, h_sol, h_fit, h_curve, wrap, n_checkpoints)

            # CS
            cs = cuckoo_search(wrap, wrap.dim, pop_size, max_iter, wrap.lb, wrap.ub)
            c_sol, c_fit, c_curve = cs.solve()
            _record(all_fits, all_sols, all_feas, all_curves,
                    'CS', run, c_sol, c_fit, c_curve, wrap, n_checkpoints)

            # HybridHOACS
            hy = HybridHOACS(wrap, wrap.dim, pop_size, max_iter, wrap.lb, wrap.ub)
            hy_sol, hy_fit, hy_curve = hy.solve()
            _record(all_fits, all_sols, all_feas, all_curves,
                    'HybridHOACS', run, hy_sol, hy_fit, hy_curve, wrap, n_checkpoints)

            print(f'HOA={all_fits["HOA"][run]:.4f}  '
                  f'CS={all_fits["CS"][run]:.4f}  '
                  f'Hybrid={all_fits["HybridHOACS"][run]:.4f}')

        for algo in ALGO_NAMES:
            f = all_fits[algo];  v = all_feas[algo]
            results[algo][pname] = {
                'errors'          : f,
                'solutions'       : all_sols[algo],
                'feasible'        : v,
                'curve'           : all_curves[algo].mean(axis=0),
                'mean'            : f.mean(),
                'std'             : f.std(),
                'best'            : f.min(),
                'worst'           : f.max(),
                'feasibility_rate': v.mean()*100.0,
            }

        _print_summary(results, pname, wrap.best_known)
        _save_npy(results, pname, save_dir)

    return results


def _record(all_fits, all_sols, all_feas, all_curves,
            algo, run, sol, fit, curve, wrap, n_cp):
    sol = np.asarray(sol).flatten()
    all_fits  [algo][run] = float(fit)
    all_sols  [algo][run] = sol
    all_feas  [algo][run] = wrap.is_feasible(sol)
    all_curves[algo][run] = _resample(np.asarray(curve, dtype=float), n_cp)


def load_results(problem_names=ALL_PROBLEMS, save_dir=SAVE_DIR) -> dict:
    results = {a: {} for a in ALGO_NAMES}
    for algo in ALGO_NAMES:
        for pname in problem_names:
            fp  = Path(save_dir) / f'eng_{algo}_{pname}_fits.npy'
            cp  = Path(save_dir) / f'eng_{algo}_{pname}_curve.npy'
            sp  = Path(save_dir) / f'eng_{algo}_{pname}_sols.npy'
            fap = Path(save_dir) / f'eng_{algo}_{pname}_feasible.npy'
            if fp.exists():
                f = np.load(fp)
                v = np.load(fap) if fap.exists() else np.ones(len(f), dtype=bool)
                results[algo][pname] = {
                    'errors'          : f,
                    'solutions'       : np.load(sp) if sp.exists() else None,
                    'feasible'        : v,
                    'curve'           : np.load(cp) if cp.exists() else np.zeros(N_CHECKPOINTS),
                    'mean'            : f.mean(), 'std': f.std(),
                    'best'            : f.min(),  'worst': f.max(),
                    'feasibility_rate': v.mean()*100.0,
                }
            else:
                print(f'[warn] missing: {fp}')
    return results


def print_results_table(results: dict, problem_names=ALL_PROBLEMS,
                        out_file: str = None):
    lines = [
        '\n' + '═'*114,
        f'  Engineering Problems — Results  ({N_RUNS} runs, {MAX_FES:,} FEs, pop={POP_SIZE})',
        '═'*114,
        f"{'Problem':<22} {'Algorithm':<16}"
        f"{'Best':>12} {'Mean':>14} {'Std':>12} {'Worst':>12} {'Feasible%':>10} {'vs Known':>10}",
        '─'*114,
    ]
    for pname in problem_names:
        prob  = get_problem(pname)
        means = [results[a][pname]['mean'] for a in ALGO_NAMES]
        best_k = int(np.argmin(means))
        for k, algo in enumerate(ALGO_NAMES):
            d      = results[algo][pname]
            marker = '*' if k == best_k else ' '
            label  = pname.replace('_',' ').title() if k == 0 else ''
            lines.append(
                f"{label:<22} {marker}{algo:<15}"
                f"{d['best']:>12.5f} {d['mean']:>14.5f} {d['std']:>12.5f} "
                f"{d['worst']:>12.5f} {d['feasibility_rate']:>9.1f}%"
                f"{d['best']-prob.best_known:>+11.5f}"
            )
        lines.append('─'*114)
    lines += ['  * = best mean   |   "vs Known" = best found − published best', '═'*114]
    _out('\n'.join(lines), out_file)


def print_best_solutions(results: dict, problem_names=ALL_PROBLEMS,
                         out_file: str = None):
    VAR_NAMES = {
        'welded_beam':     ['h (weld thick.in)', 'l (weld len.in)',
                            't (bar height.in)', 'b (bar thick.in)'],
        'pressure_vessel': ['Ts (shell thick.in)', 'Th (head thick.in)',
                            'R (radius in)',       'L (length in)'],
        'spring_design':   ['d (wire dia.)', 'D (coil dia.)', 'N (active coils)'],
        'speed_reducer':   ['b (face width)', 'm (module)', 'z (teeth)',
                            'l1 (shaft1 len.)', 'l2 (shaft2 len.)',
                            'd1 (shaft1 dia.)', 'd2 (shaft2 dia.)'],
        'three_bar_truss': ['A1 (bars 1&3 cm²)', 'A2 (bar 2 cm²)'],
    }
    lines = ['\n'+'═'*62, '  Best Solutions Found by HybridHOACS', '═'*62]
    for pname in problem_names:
        prob     = get_problem(pname)
        d        = results['HybridHOACS'][pname]
        best_run = int(np.argmin(d['errors']))
        best_x   = d['solutions'][best_run] if d['solutions'] is not None \
                   else np.zeros(prob.dim)
        lines += [f'\n  {prob.name}', '  '+'─'*44,
                  f'  Best fitness : {d["errors"][best_run]:.6f}',
                  f'  Known best   : {prob.best_known:.6f}',
                  f'  Gap          : {d["errors"][best_run]-prob.best_known:+.6f}',
                  f'  Feasible     : {"Yes" if d["feasible"][best_run] else "No"}',
                  '  Variables:']
        for vn, val in zip(VAR_NAMES.get(pname,[f'x{i}' for i in range(prob.dim)]), best_x):
            lines.append(f'    {vn:<30}: {val:.6f}')
    lines.append('\n'+'═'*62)
    _out('\n'.join(lines), out_file)


def print_wilcoxon_table(results: dict, proposed='HybridHOACS',
                         problem_names=ALL_PROBLEMS, alpha=ALPHA,
                         out_file: str = None):
    competitors = [a for a in ALGO_NAMES if a != proposed]
    col    = 16
    header = f"{'Problem':<26}" + ''.join(f'{c:^{col}}' for c in competitors)
    summary = {c: {'+':0,'-':0,'=':0} for c in competitors}
    lines   = ['\n'+'═'*len(header),
               f'  Wilcoxon Rank-Sum  |  proposed={proposed}  |  α={alpha}',
               '═'*len(header), header, '─'*len(header)]

    for pname in problem_names:
        prop_e = results[proposed][pname]['errors']
        row    = f'{pname.replace("_"," ").title():<26}'
        for c in competitors:
            stat, p = ranksums(results[c][pname]['errors'], prop_e)
            sym = ('=' if p >= alpha else ('+' if stat > 0 else '-'))
            summary[c][sym] += 1
            row += f'{sym:^{col}}'
        lines.append(row)

    lines.append('─'*len(header))
    for label, key in [('(+) better','+'),('(-) worse','-'),('(=) similar','=')]:
        lines.append(f'{label:<26}'+''.join(f'{summary[c][key]:^{col}}' for c in competitors))
    lines.append('═'*len(header))
    _out('\n'.join(lines), out_file)


def plot_convergence_curves(results: dict, problem_names=ALL_PROBLEMS,
                            max_fes=MAX_FES, save_dir=FIG_DIR):
    os.makedirs(save_dir, exist_ok=True)
    x_axis = np.linspace(max_fes/N_CHECKPOINTS, max_fes, N_CHECKPOINTS)

    for pname in problem_names:
        prob = get_problem(pname)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for algo, sty in STYLES.items():
            if pname not in results.get(algo, {}):
                continue
            c = np.clip(results[algo][pname]['curve'], 1e-300, None)
            ax.semilogy(x_axis, c, label=algo,
                        color=sty['color'], linewidth=sty['lw'],
                        dashes=sty['dash'] or [])
        ax.axhline(prob.best_known, color='#888780', lw=1.0,
                   linestyle=':', label=f'Known best ({prob.best_known:.4f})')
        ax.set_xlabel('Function Evaluations (FEs)', fontsize=11)
        ax.set_ylabel('Best Fitness (log scale)', fontsize=11)
        ax.set_title(prob.name, fontsize=12)
        ax.legend(fontsize=9, framealpha=0.35, edgecolor='none')
        ax.grid(True, which='both', alpha=0.18, linewidth=0.5)
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        plt.tight_layout()
        path = f'{save_dir}/{pname}_convergence.pdf'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_panel(results: dict, problem_names=ALL_PROBLEMS,
               max_fes=MAX_FES, save_dir=FIG_DIR):
    """2×3 panel — all 5 problems in one figure for the paper body."""
    os.makedirs(save_dir, exist_ok=True)
    x_axis     = np.linspace(max_fes/N_CHECKPOINTS, max_fes, N_CHECKPOINTS)
    cols, rows = 3, 2
    fig, axes  = plt.subplots(rows, cols, figsize=(cols*3.8, rows*3.2))
    axes       = np.array(axes).flatten()

    for idx, pname in enumerate(problem_names):
        ax   = axes[idx]
        prob = get_problem(pname)
        for algo, sty in STYLES.items():
            if pname not in results.get(algo, {}):
                continue
            c = np.clip(results[algo][pname]['curve'], 1e-300, None)
            ax.semilogy(x_axis, c,
                        label=algo if idx == 0 else '',
                        color=sty['color'], linewidth=sty['lw'],
                        dashes=sty['dash'] or [])
        ax.axhline(prob.best_known, color='#888780', lw=0.9, linestyle=':',
                   label='Known best' if idx == 0 else '')
        ax.set_title(prob.name, fontsize=9, pad=3)
        ax.grid(True, which='both', alpha=0.15, lw=0.4)
        ax.tick_params(labelsize=7)
        if idx % cols == 0: ax.set_ylabel('Fitness', fontsize=8)
        if idx >= (rows-1)*cols: ax.set_xlabel('FEs', fontsize=8)

    for ax in axes[len(problem_names):]: ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(STYLES)+1,
               fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle('Engineering Problems — Convergence Curves', y=1.05, fontsize=12)
    plt.tight_layout()
    path = f'{save_dir}/engineering_panel.pdf'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved panel: {path}')

def _resample(arr, n):
    if len(arr) == n: return arr.copy()
    return np.interp(np.linspace(0,1,n), np.linspace(0,1,len(arr)), arr)

def _out(text, out_file=None):
    print(text)
    if out_file:
        with open(out_file, 'a') as f: f.write(text+'\n\n')

def _print_summary(results, pname, known):
    print(f'\n  ── {pname.replace("_"," ").title()} ──')
    for a in ALGO_NAMES:
        d = results[a][pname]
        print(f'  {a:<14} mean={d["mean"]:.4f}  std={d["std"]:.4f}  '
              f'best={d["best"]:.4f}  feasible={d["feasibility_rate"]:.0f}%')
    print()

def _save_npy(results, pname, save_dir):
    for algo in ALGO_NAMES:
        if pname not in results[algo]: continue
        d    = results[algo][pname]
        stem = Path(save_dir) / f'eng_{algo}_{pname}'
        np.save(f'{stem}_fits.npy',     d['errors'])
        np.save(f'{stem}_curve.npy',    d['curve'])
        np.save(f'{stem}_feasible.npy', d['feasible'])
        if d['solutions'] is not None:
            np.save(f'{stem}_sols.npy', d['solutions'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',   action='store_true')
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()

    if args.test:
        quick_test(); sys.exit(0)

    os.makedirs(SAVE_DIR, exist_ok=True)
    OUT = Path(SAVE_DIR) / 'engineering_report.txt'
    with open(OUT, 'w') as f:
        f.write('HHOCS Engineering Benchmark Results\n' + '='*60 + '\n\n')

    if args.reload:
        print('Reloading saved results...')
        results = load_results()
    else:
        print('='*62)
        print(f'  Engineering Benchmark  |  {len(ALL_PROBLEMS)} problems')
        print(f'  {N_RUNS} runs × {MAX_FES:,} FEs  |  pop={POP_SIZE}')
        print('='*62)
        results = run_full_experiment()

    print_results_table(results,  out_file=str(OUT))
    print_best_solutions(results, out_file=str(OUT))
    print_wilcoxon_table(results, out_file=str(OUT))

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_convergence_curves(results)
    plot_panel(results)
