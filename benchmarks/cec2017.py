"""
CEC2017 Benchmark Functions (F1 - F29)
Based on: Awad et al., "Problem Definitions and Evaluation Criteria for the
CEC 2017 Special Session and Competition on Single Objective Real-Parameter
Numerical Optimization", 2016.

Usage — standalone (no data files needed)
-----------------------------------------
    from cec2017 import get_function, FUNCTION_INFO

    f9 = get_function(9, dim=10)
    print(f9([0]*10))           # near bias (900)

    f9 = get_function(9, dim=10, shift=True, rotate=True, seed=42)
    print(f9.evaluate([0]*10))

Usage — with official data files
---------------------------------
    f9 = get_function(9, dim=10, data_dir='cec2017_data')
    print(f9([0]*10))

    Download data: https://github.com/P-N-Suganthan/CEC2017
    Place in:      ./cec2017_data/

Notes
-----
* Biases: F1=100, F2=200, ..., F29=2900
* All functions are minimisation problems.
* Search range: [-100, 100]^D
* Valid dims: 10, 20, 30, 50, 100
"""

import numpy as np
from pathlib import Path


# =============================================================================
# Shared helpers  (same pattern as cec2014.py)
# =============================================================================

def _make_shift(dim, lb=-80, ub=80, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(lb, ub, dim)


def _make_rotation(dim, seed=None):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return Q


# =============================================================================
# Base component functions  (operate on already-transformed z)
# =============================================================================

def _elliptic(z):
    D = len(z)
    if D == 1:
        return float(z[0] ** 2)
    exp = np.arange(D) / (D - 1.0)
    return float(np.sum((1e6 ** exp) * z ** 2))

def _bent_cigar(z):
    return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))

def _discus(z):
    return float(1e6 * z[0] ** 2 + np.sum(z[1:] ** 2))

def _rosenbrock(z):
    w = z + 1.0
    return float(np.sum(100.0 * (w[:-1] ** 2 - w[1:]) ** 2 + (w[:-1] - 1.0) ** 2))

def _ackley(z):
    D = len(z)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2)))
                 - np.exp(np.mean(np.cos(2.0 * np.pi * z))) + 20.0 + np.e)

def _weierstrass(z):
    a, b, kmax = 0.5, 3.0, 20
    k  = np.arange(kmax + 1)
    ak = a ** k;  bk = b ** k
    s1 = sum(np.sum(ak * np.cos(2.0 * np.pi * bk * (xi + 0.5))) for xi in z)
    s2 = len(z) * np.sum(ak * np.cos(np.pi * bk))
    return float(s1 - s2)

def _griewank(z):
    idx = np.arange(1, len(z) + 1)
    return float(np.sum(z ** 2) / 4000.0 - np.prod(np.cos(z / np.sqrt(idx))) + 1.0)

def _rastrigin(z):
    return float(np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))

def _modified_schwefel(z):
    D = len(z);  s = 0.0
    for zi in z:
        xi = zi + 4.209687462275036e2
        if abs(xi) <= 500.0:
            s -= xi * np.sin(np.sqrt(abs(xi)))
        elif xi > 500.0:
            s += (xi - 500.0) ** 2 / 100.0 - 500.0 * np.sin(np.sqrt(abs(xi)))
        else:
            s += (-xi - 500.0) ** 2 / 100.0 - 500.0 * np.sin(np.sqrt(abs(-xi)))
    return float(s + 4.189828872724338e2 * D)

def _katsuura(z):
    D = len(z);  prod = 1.0
    for i, xi in enumerate(z):
        s = sum(abs(2**j * xi - round(2**j * xi)) / 2**j for j in range(1, 33))
        prod *= (1.0 + (i + 1) * s) ** (10.0 / D ** 1.2)
    return float((10.0 / D ** 2) * (prod - 1.0))

def _happycat(z):
    D = len(z);  s2 = np.sum(z**2);  s1 = np.sum(z)
    return float(abs(s2 - D) ** 0.25 + (0.5 * s2 + s1) / D + 0.5)

def _hgbat(z):
    D = len(z);  s2 = np.sum(z**2);  s1 = np.sum(z)
    return float(abs(s2**2 - s1**2) ** 0.5 + (0.5 * s2 + s1) / D + 0.5)

def _expanded_scaffer(z):
    D = len(z)
    def _f6(a, b):
        t = a**2 + b**2
        return 0.5 + (np.sin(np.sqrt(t))**2 - 0.5) / (1.0 + 0.001 * t)**2
    return float(sum(_f6(z[i], z[(i+1) % D]) for i in range(D)))

def _expanded_griewank_rosenbrock(z):
    D = len(z);  total = 0.0
    for i in range(D):
        j   = (i + 1) % D
        tmp = 100.0 * (z[i]**2 - z[j])**2 + (z[i] - 1.0)**2
        total += tmp**2 / 4000.0 - np.cos(tmp) + 1.0
    return float(total)

def _levy(z):
    w  = 1.0 + (z - 1.0) / 4.0
    s  = np.sin(np.pi * w[0])**2
    s += np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0)**2))
    s += (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[-1])**2)
    return float(s)

def _zakharov(z):
    idx = np.arange(1, len(z) + 1)
    s1  = np.sum(z**2);  s2 = np.sum(0.5 * idx * z)
    return float(s1 + s2**2 + s2**4)

# Internal registry for hybrid/composition lookups
_BASE = {
    'elliptic':        _elliptic,
    'bent_cigar':      _bent_cigar,
    'discus':          _discus,
    'rosenbrock':      _rosenbrock,
    'ackley':          _ackley,
    'weierstrass':     _weierstrass,
    'griewank':        _griewank,
    'rastrigin':       _rastrigin,
    'schwefel':        _modified_schwefel,
    'modified_schwefel': _modified_schwefel,
    'katsuura':        _katsuura,
    'happycat':        _happycat,
    'hgbat':           _hgbat,
    'scaffer':         _expanded_scaffer,
    'expanded_scaffer':_expanded_scaffer,
    'grie_rosen':      _expanded_griewank_rosenbrock,
    'levy':            _levy,
    'zakharov':        _zakharov,
}


# =============================================================================
# Hybrid function component definitions  (F11–F20)
# Each entry: list of (base_fn_name, proportion)
# =============================================================================

_HYBRID_DEFS = {
    11: [('zakharov',   0.2), ('rosenbrock', 0.4), ('elliptic',    0.4)],
    12: [('elliptic',   0.3), ('schwefel',   0.3), ('bent_cigar',  0.4)],
    13: [('bent_cigar', 0.3), ('rosenbrock', 0.3), ('schwefel',    0.4)],
    14: [('elliptic',   0.2), ('ackley',     0.2), ('schwefel',    0.3), ('rastrigin', 0.3)],
    15: [('bent_cigar', 0.1), ('hgbat',      0.2), ('rastrigin',   0.3), ('rosenbrock',0.4)],
    16: [('schwefel',   0.2), ('ackley',     0.2), ('rosenbrock',  0.3), ('happycat',  0.3)],
    17: [('elliptic',   0.1), ('ackley',     0.2), ('rastrigin',   0.2),
         ('hgbat',      0.2), ('discus',     0.3)],
    18: [('bent_cigar', 0.1), ('rastrigin',  0.2), ('schwefel',    0.3),
         ('happycat',   0.2), ('elliptic',   0.2)],
    19: [('elliptic',   0.1), ('ackley',     0.1), ('rastrigin',   0.2), ('schwefel',  0.2),
         ('hgbat',      0.2), ('rosenbrock', 0.2)],
    20: [('ackley',     0.1), ('elliptic',   0.1), ('rastrigin',   0.2), ('schwefel',  0.2),
         ('hgbat',      0.1), ('rosenbrock', 0.1), ('discus',      0.1), ('happycat',  0.1)],
}


# =============================================================================
# Composition function component definitions  (F21–F29)
# Each entry: list of (base_fn_name, sigma_i, lambda_i, bias_i)
# =============================================================================

_COMP_DEFS = {
    21: [('rosenbrock', 10,  2.5e-4,   0), ('elliptic',  20, 1e-6,  100),
         ('bent_cigar', 30,  10,      200), ('discus',    40, 1e-6,  300),
         ('elliptic',   50,  1e-6,   400)],
    22: [('rastrigin',  10,  0.1,      0), ('happycat',  20, 0.2,   100),
         ('ackley',     30,  0.3,    200), ('discus',    40, 0.5,   300),
         ('rosenbrock', 50,  0.1,   400)],
    23: [('rosenbrock', 10,  2.5e-4,   0), ('ackley',    20, 10,    100),
         ('schwefel',   30,  1,      200), ('happycat',  40, 5e-4,  300),
         ('elliptic',   50,  1e-6,   400)],
    24: [('ackley',     10,  10,       0), ('schwefel',  20, 1,     100),
         ('griewank',   30,  10,     200), ('rosenbrock',40, 2.5e-4,300),
         ('rastrigin',  50,  0.1,   400)],
    25: [('rastrigin',  10,  0.1,      0), ('happycat',  20, 0.5,   100),
         ('ackley',     30,  2,      200), ('griewank',  40, 10,    300),
         ('rosenbrock', 50,  2.5e-4, 400)],
    26: [('scaffer',    10,  0.5,      0), ('schwefel',  20, 1,     100),
         ('griewank',   30,  10,     200), ('rosenbrock',40, 2.5e-4,300),
         ('rastrigin',  50,  0.1,   400)],
    27: [('hgbat',      10,  10,       0), ('rastrigin', 20, 0.1,   100),
         ('schwefel',   30,  1,      200), ('bent_cigar',40, 1e-6,  300),
         ('elliptic',   50,  1e-6,  400)],
    28: [('ackley',     10,  10,       0), ('griewank',  20, 10,    100),
         ('discus',     30,  1e-6,   200), ('rosenbrock',40, 2.5e-4,300),
         ('happycat',   50,  0.5,   400)],
    29: [('schwefel',   10,  1,        0), ('ackley',    20, 10,    100),
         ('rosenbrock', 30,  2.5e-4,  200), ('happycat', 40, 0.5,  300),
         ('rastrigin',  50,  0.1,   400)],
}


# =============================================================================
# CEC2017Function class  (mirrors CEC2014Function interface)
# =============================================================================

class CEC2017Function:
    """
    Wraps a CEC2017 benchmark function.

    Parameters
    ----------
    func_id  : int 1–29
    dim      : int problem dimension
    shift    : bool or ndarray
    rotate   : bool or ndarray
    seed     : int  RNG seed (used when shift/rotate are True)
    data_dir : str  path to official data files (overrides shift/rotate if given)
    """

    BIASES = {i: i * 100 for i in range(1, 30)}

    def __init__(self, func_id, dim=10, shift=False, rotate=False,
                 seed=None, data_dir=None):
        if not 1 <= func_id <= 29:
            raise ValueError("func_id must be between 1 and 29.")
        self.func_id = func_id
        self.dim     = dim
        self.bias    = self.BIASES[func_id]
        self._use_official = data_dir is not None

        if self._use_official:
            self._load_official(data_dir)
        else:
            # Standalone mode — random shift/rotation like cec2014.py
            self.o = (shift if isinstance(shift, np.ndarray)
                      else (_make_shift(dim, seed=seed) if shift
                            else np.zeros(dim)))
            self.M = (rotate if isinstance(rotate, np.ndarray)
                      else (_make_rotation(dim, seed=seed) if rotate
                            else np.eye(dim)))
            # Per-component rotation matrices for composition functions
            if func_id >= 21:
                n_comp = len(_COMP_DEFS[func_id])
                rng    = np.random.default_rng(func_id * 1000)
                self._comp_shifts = [rng.uniform(-80, 80, dim) for _ in range(n_comp)]
                self._comp_Ms     = [_make_rotation(dim, seed=func_id*1000+k)
                                     for k in range(n_comp)]
            else:
                self._comp_shifts = []
                self._comp_Ms     = []

        self.S = None   # shuffle vector — only used with official data files

    def _load_official(self, data_dir):
        ddir = Path(data_dir)
        fid, D = self.func_id, self.dim

        def _ld(fname):
            p = ddir / fname
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing CEC2017 data file: {p}\n"
                    f"Download from: https://github.com/P-N-Suganthan/CEC2017\n"
                    f"Place all files in: {ddir}/")
            return np.loadtxt(p)

        self.o = _ld(f'shift_data_{fid}.txt').flatten()[:D]
        self.M = _ld(f'M_{fid}_D{D}.txt')

        self.S = None
        if fid >= 11:
            shuf   = _ld(f'shuffle_data_{fid}_D{D}.txt').flatten().astype(int)
            self.S = shuf[:D] - 1

        self._comp_Ms     = []
        self._comp_shifts = []
        if fid >= 21:
            n_comp = len(_COMP_DEFS[fid])
            for k in range(1, n_comp + 1):
                fname_k = f'M_{fid}_c{k}_D{D}.txt'
                fname_s = f'M_{fid}_D{D}.txt'
                p_k = ddir / fname_k
                self._comp_Ms.append(_ld(fname_k if p_k.exists() else fname_s))
            # shifts for composition: successive slices of o
            for i in range(n_comp):
                start = (i * D) % len(self.o)
                self._comp_shifts.append(np.roll(self.o, -start)[:D])

    # ------------------------------------------------------------------
    def _transform(self, x):
        return self.M @ (np.asarray(x, dtype=float) - self.o)

    def evaluate(self, x):
        x   = np.asarray(x, dtype=float)
        fid = self.func_id
        z   = self._transform(x)

        # Unimodal F1–F3
        if   fid == 1: return self.bias + _elliptic(z)
        elif fid == 2: return self.bias + _bent_cigar(z)
        elif fid == 3: return self.bias + _discus(z)

        # Simple multimodal F4–F10
        elif fid == 4:  return self.bias + _rosenbrock(z)
        elif fid == 5:  return self.bias + _ackley(z)
        elif fid == 6:  return self.bias + _weierstrass(z)
        elif fid == 7:  return self.bias + _griewank(z)
        elif fid == 8:  return self.bias + _modified_schwefel(z)
        elif fid == 9:  return self.bias + _rastrigin(z)
        elif fid == 10: return self.bias + _expanded_scaffer(z)

        # Hybrid F11–F20
        elif 11 <= fid <= 20:
            return self.bias + self._eval_hybrid(x, fid)

        # Composition F21–F29
        elif 21 <= fid <= 29:
            return self.bias + self._eval_composition(x, fid)

    def _eval_hybrid(self, x, fid):
        z_rot  = self.M @ (x - self.o)
        z_shuf = z_rot[self.S] if self.S is not None else z_rot
        comps  = _HYBRID_DEFS[fid]
        total, start = 0.0, 0
        for fn_name, prop in comps:
            size  = max(1, int(round(prop * self.dim)))
            end   = min(start + size, self.dim)
            total += _BASE[fn_name](z_shuf[start:end])
            start  = end
            if start >= self.dim:
                break
        return total

    def _eval_composition(self, x, fid):
        D     = self.dim
        comps = _COMP_DEFS[fid]
        N     = len(comps)
        w     = np.zeros(N)
        fit   = np.zeros(N)

        for i, (fn_name, sigma_i, lambda_i, bias_i) in enumerate(comps):
            o_i  = self._comp_shifts[i] if i < len(self._comp_shifts) else self.o
            M_i  = self._comp_Ms[i]     if i < len(self._comp_Ms)     else self.M
            z_i  = x - o_i
            z_r  = M_i @ z_i
            fit[i] = _BASE[fn_name](z_r / lambda_i) + bias_i
            w[i]   = np.exp(-np.sum(z_i**2) / (2.0 * D * sigma_i**2))

        w_max = w.max()
        if w_max == 0.0:
            w[:] = 1.0
        else:
            mask = w != w_max
            w[mask] *= 1.0 / (1.0 + 1e-6 * (w_max / np.maximum(w[mask], 1e-300)))

        w_sum = w.sum()
        w = w / w_sum if w_sum > 0 else np.full(N, 1.0 / N)
        return float(np.dot(w, fit))

    def __call__(self, x):
        return self.evaluate(x)

    def __repr__(self):
        return (f"CEC2017 F{self.func_id} | dim={self.dim} | bias={self.bias} | "
                f"shift={'custom' if np.any(self.o) else 'none'} | "
                f"rotate={'yes' if not np.array_equal(self.M, np.eye(self.dim)) else 'no'}")


# =============================================================================
# Factory  (mirrors cec2014.py get_function)
# =============================================================================

def get_function(func_id, dim=10, shift=False, rotate=False,
                 seed=None, data_dir=None):
    """
    Returns a callable CEC2017Function.

    Example
    -------
        f9 = get_function(9, dim=10, shift=True, rotate=True, seed=42)
        loss = f9(x)
    """
    return CEC2017Function(func_id, dim=dim, shift=shift, rotate=rotate,
                           seed=seed, data_dir=data_dir)


# =============================================================================
# Benchmark runner  (mirrors cec2014.py run_benchmark)
# =============================================================================

def run_benchmark(solver_factory, func_ids=range(1, 30), dim=10,
                  shift=True, rotate=True, seed=42, verbose=True):
    """
    Runs a solver against a set of CEC2017 functions.

    Parameters
    ----------
    solver_factory : callable
        Called as solver_factory(obj_func, dim), must return object
        with .solve() → (best_sol, best_fit, curve)

    Example
    -------
        from cec2017 import run_benchmark
        from algorithms.hybrid import HybridHOACS

        def make_solver(obj, dim):
            return HybridHOACS(obj, dim, pop_size=30, max_iter=100, lb=-100, ub=100)

        results = run_benchmark(make_solver, func_ids=range(1, 11), dim=10)
    """
    results = {}
    for fid in func_ids:
        f = get_function(fid, dim=dim, shift=shift, rotate=rotate, seed=seed)
        solver = solver_factory(f, dim)
        _, best_fit, curve = solver.solve()
        results[fid] = (best_fit, curve)
        if verbose:
            print(f"  F{fid:02d}  best={best_fit:.6f}  bias={f.bias}"
                  f"  error={best_fit - f.bias:.6f}")
    return results


# =============================================================================
# Metadata  (mirrors cec2014.py FUNCTION_INFO)
# =============================================================================

FUNCTION_INFO = {
    1:  ("F01",  "Shifted and Rotated Bent Cigar",                    "Unimodal"),
    2:  ("F02",  "Shifted and Rotated Sum of Different Power",        "Unimodal"),
    3:  ("F03",  "Shifted and Rotated Zakharov",                      "Unimodal"),
    4:  ("F04",  "Shifted and Rotated Rosenbrock's",                  "Multimodal"),
    5:  ("F05",  "Shifted and Rotated Rastrigin's",                   "Multimodal"),
    6:  ("F06",  "Shifted and Rotated Expanded Scaffer's F6",         "Multimodal"),
    7:  ("F07",  "Shifted and Rotated Lunacek Bi-Rastrigin",          "Multimodal"),
    8:  ("F08",  "Shifted and Rotated Non-Cont. Rastrigin's",         "Multimodal"),
    9:  ("F09",  "Shifted and Rotated Levy",                          "Multimodal"),
    10: ("F10",  "Shifted and Rotated Modified Schwefel's",           "Multimodal"),
    11: ("F11",  "Hybrid Function 1 (N=3)",                           "Hybrid"),
    12: ("F12",  "Hybrid Function 2 (N=3)",                           "Hybrid"),
    13: ("F13",  "Hybrid Function 3 (N=3)",                           "Hybrid"),
    14: ("F14",  "Hybrid Function 4 (N=4)",                           "Hybrid"),
    15: ("F15",  "Hybrid Function 5 (N=4)",                           "Hybrid"),
    16: ("F16",  "Hybrid Function 6 (N=4)",                           "Hybrid"),
    17: ("F17",  "Hybrid Function 7 (N=5)",                           "Hybrid"),
    18: ("F18",  "Hybrid Function 8 (N=5)",                           "Hybrid"),
    19: ("F19",  "Hybrid Function 9 (N=6)",                           "Hybrid"),
    20: ("F20",  "Hybrid Function 10 (N=8)",                          "Hybrid"),
    21: ("F21",  "Composition Function 1 (N=5)",                      "Composition"),
    22: ("F22",  "Composition Function 2 (N=5)",                      "Composition"),
    23: ("F23",  "Composition Function 3 (N=5)",                      "Composition"),
    24: ("F24",  "Composition Function 4 (N=5)",                      "Composition"),
    25: ("F25",  "Composition Function 5 (N=5)",                      "Composition"),
    26: ("F26",  "Composition Function 6 (N=5)",                      "Composition"),
    27: ("F27",  "Composition Function 7 (N=5)",                      "Composition"),
    28: ("F28",  "Composition Function 8 (N=5)",                      "Composition"),
    29: ("F29",  "Composition Function 9 (N=5)",                      "Composition"),
}