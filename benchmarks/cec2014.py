"""
CEC2014 Benchmark Functions (F1 - F30)
Based on: Liang et al., "Problem Definitions and Evaluation Criteria for the
CEC 2014 Special Session and Competition on Single Objective Real-Parameter
Numerical Optimization", 2013.

Usage
-----
    from cec2014 import get_function, CEC2014Function

    # Quick usage — no shift/rotation (zero-shifted, identity rotation)
    f1 = get_function(1, dim=10)
    print(f1([0]*10))   # should be near 100.0 (the bias)

    # With random shift and rotation (recommended for fair benchmarking)
    f1 = get_function(1, dim=10, shift=True, rotate=True, seed=42)
    print(f1.evaluate([0]*10))

Notes
-----
* The official CEC2014 suite requires proprietary .mat shift/rotation data files
  distributed by the competition organisers.  This file re-implements the
  mathematical formulations from scratch so it works standalone.
* Biases match the original paper: F1=100, F2=200, ..., F30=3000.
* All functions are minimisation problems.
* Recommended search range: [-100, 100]^D for all functions.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shift(dim, lb=-80, ub=80, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(lb, ub, dim)


def _make_rotation(dim, seed=None):
    """Random orthogonal rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    return Q


def _shifted(x, o):
    return np.asarray(x, dtype=float) - o


def _rotated(z, M):
    return M @ z


# ---------------------------------------------------------------------------
# Core component functions  (operate on already-transformed z)
# ---------------------------------------------------------------------------

def _sphere(z):
    return np.sum(z ** 2)

def _elliptic(z):
    dim = len(z)
    if dim == 1:
        return z[0] ** 2
    exponents = np.arange(dim) / (dim - 1)
    return np.sum((1e6 ** exponents) * z ** 2)

def _bent_cigar(z):
    return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)

def _discus(z):
    return 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2)

def _rosenbrock(z):
    """Shifted so minimum is near origin: use z+1."""
    w = z + 1
    return np.sum(100 * (w[1:] - w[:-1] ** 2) ** 2 + (w[:-1] - 1) ** 2)

def _ackley(z):
    dim = len(z)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / dim))
            - np.exp(np.sum(np.cos(2 * np.pi * z)) / dim)
            + 20 + np.e)

def _weierstrass(z, kmax=20):
    a, b = 0.5, 3.0
    k = np.arange(kmax + 1)
    ak = a ** k
    bk = b ** k
    sum1 = np.sum([np.sum(ak * np.cos(2 * np.pi * bk * (zi + 0.5))) for zi in z])
    dim = len(z)
    sum2 = dim * np.sum(ak * np.cos(np.pi * bk))
    return sum1 - sum2

def _griewank(z):
    dim = len(z)
    idx = np.arange(1, dim + 1)
    return np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(idx))) + 1

def _rastrigin(z):
    return 10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

def _modified_schwefel(z):
    """CEC2014 variant of Schwefel."""
    dim = len(z)
    w = z + 4.209687462275036e2  # shift so min is at 0
    val = 0.0
    for wi in w:
        awi = abs(wi)
        if awi <= 500:
            val -= wi * np.sin(np.sqrt(awi))
        elif wi > 500:
            val += 0.001 * (wi - 500) ** 2 - wi * np.sin(np.sqrt(abs(wi)))
        else:
            val += 0.001 * (wi + 500) ** 2 - wi * np.sin(np.sqrt(abs(wi)))
    return val + 4.189828872724338e2 * dim

def _katsuura(z):
    dim = len(z)
    prod = 1.0
    for i, zi in enumerate(z):
        inner = 0.0
        for j in range(1, 33):
            t = 2 ** j * zi
            inner += abs(t - np.round(t)) / (2 ** j)
        prod *= (1 + (i + 1) * inner) ** (10 / dim ** 1.2)
    return (10 / dim ** 2) * (prod - 1)

def _happy_cat(z):
    dim = len(z)
    norm_sq = np.sum(z**2)
    return abs(norm_sq - dim) ** 0.25 + (0.5 * norm_sq + np.sum(z)) / dim + 0.5

def _hgbat(z):
    dim = len(z)
    norm_sq = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(norm_sq**2 - sum_z**2) ** 0.5 + (0.5 * norm_sq + sum_z) / dim + 0.5

def _expanded_griewank_rosenbrock(z):
    """EGR: Griewank applied to Rosenbrock chain."""
    dim = len(z)
    w = z + 1
    total = 0.0
    for i in range(dim - 1):
        t = 100 * (w[i] ** 2 - w[i+1]) ** 2 + (w[i] - 1) ** 2
        total += t ** 2 / 4000 - np.cos(t) + 1
    # wrap-around
    t = 100 * (w[-1] ** 2 - w[0]) ** 2 + (w[-1] - 1) ** 2
    total += t ** 2 / 4000 - np.cos(t) + 1
    return total

def _expanded_scaffer(z):
    """Expanded Scaffer's F6."""
    dim = len(z)
    def scaffer_f6(a, b):
        num = np.sin(np.sqrt(a**2 + b**2))**2 - 0.5
        den = (1 + 0.001 * (a**2 + b**2))**2
        return 0.5 + num / den
    total = 0.0
    for i in range(dim - 1):
        total += scaffer_f6(z[i], z[i+1])
    total += scaffer_f6(z[-1], z[0])
    return total


# ---------------------------------------------------------------------------
# Hybrid / Composition helpers
# ---------------------------------------------------------------------------

def _lambda_scale(z, lam):
    """Scale z so each component function operates at similar magnitude."""
    return z * lam


def _composition_weights(z_list, sigmas, lambdas, funcs, biases):
    """
    Generic composition function used in F21–F30.
    z_list : list of already-shifted+rotated vectors, one per component
    """
    n = len(funcs)
    w = np.zeros(n)
    for i in range(n):
        norm_sq = np.sum(z_list[i] ** 2)
        w[i] = np.exp(-norm_sq / (2 * len(z_list[i]) * sigmas[i] ** 2))

    # Avoid all-zero weights
    if np.max(w) < 1e-300:
        w[:] = 1.0

    w_sum = np.sum(w)
    f_vals = np.array([lambdas[i] * funcs[i](z_list[i]) + biases[i] for i in range(n)])
    return np.sum(w * f_vals) / w_sum


# ---------------------------------------------------------------------------
# CEC2014Function class
# ---------------------------------------------------------------------------

class CEC2014Function:
    """
    Wraps a CEC2014 benchmark function with shift vector and rotation matrix.

    Parameters
    ----------
    func_id  : int, 1–30
    dim      : int, problem dimension (2, 10, 20, 30, 50 are standard)
    shift    : bool or ndarray — True → random shift, ndarray → use that
    rotate   : bool or ndarray — True → random rotation, ndarray → use that
    seed     : int, RNG seed (only used when shift/rotate are True)
    """

    BIASES = {i: i * 100 for i in range(1, 31)}

    def __init__(self, func_id, dim=10, shift=False, rotate=False, seed=None):
        if not 1 <= func_id <= 30:
            raise ValueError("func_id must be between 1 and 30.")
        self.func_id = func_id
        self.dim = dim
        self.bias = self.BIASES[func_id]

        # Shift vector
        if isinstance(shift, np.ndarray):
            self.o = shift
        elif shift:
            self.o = _make_shift(dim, seed=seed)
        else:
            self.o = np.zeros(dim)

        # Rotation matrix
        if isinstance(rotate, np.ndarray):
            self.M = rotate
        elif rotate:
            self.M = _make_rotation(dim, seed=seed)
        else:
            self.M = np.eye(dim)

        self._build()

    # ------------------------------------------------------------------
    def _transform(self, x):
        z = _shifted(x, self.o)
        z = _rotated(z, self.M)
        return z

    def _build(self):
        fid = self.func_id

        # Simple unimodal (F1–F5)
        if fid == 1:
            self._core = lambda z: _elliptic(z)
        elif fid == 2:
            self._core = lambda z: _bent_cigar(z)
        elif fid == 3:
            self._core = lambda z: _discus(z)
        elif fid == 4:
            self._core = lambda z: _elliptic(z) + _ackley(z)   # Shifted Rosenbrock appended
        elif fid == 5:
            self._core = lambda z: _ackley(z)

        # Multimodal (F6–F20)
        elif fid == 6:
            self._core = lambda z: _weierstrass(z / 100)
        elif fid == 7:
            self._core = lambda z: _griewank(z)
        elif fid == 8:
            self._core = lambda z: _rastrigin(z)
        elif fid == 9:
            self._core = lambda z: _rastrigin(z)          # rotated variant
        elif fid == 10:
            self._core = lambda z: _modified_schwefel(z)
        elif fid == 11:
            self._core = lambda z: _modified_schwefel(z)  # rotated
        elif fid == 12:
            self._core = lambda z: _katsuura(z / 100)
        elif fid == 13:
            self._core = lambda z: _happy_cat(z / 100)
        elif fid == 14:
            self._core = lambda z: _hgbat(z / 100)
        elif fid == 15:
            self._core = lambda z: _expanded_griewank_rosenbrock(z / 2.048)
        elif fid == 16:
            self._core = lambda z: _expanded_scaffer(z)

        # Hybrid (F17–F22) — partition the solution vector
        elif fid == 17:
            self._core = self._hybrid_2(
                [_modified_schwefel, _rastrigin],
                [0.5, 0.5], [1.0, 1.0]
            )
        elif fid == 18:
            self._core = self._hybrid_3(
                [_modified_schwefel, _rastrigin, _elliptic],
                [0.3, 0.3, 0.4], [1.0, 1.0, 1.0]
            )
        elif fid == 19:
            self._core = self._hybrid_4(
                [_rosenbrock, _ackley, _modified_schwefel, _rastrigin],
                [0.2, 0.2, 0.3, 0.3], [1.0, 1.0, 1.0, 1.0]
            )
        elif fid == 20:
            self._core = self._hybrid_4(
                [_griewank, _weierstrass, _rosenbrock, _expanded_scaffer],
                [0.2, 0.2, 0.3, 0.3], [1.0, 1.0, 1.0, 1.0]
            )
        elif fid == 21:
            self._core = self._hybrid_5(
                [_expanded_scaffer, _hgbat, _rosenbrock, _modified_schwefel, _elliptic],
                [0.1, 0.2, 0.2, 0.2, 0.3], [1.0]*5
            )
        elif fid == 22:
            self._core = self._hybrid_5(
                [_katsuura, _happy_cat, _expanded_griewank_rosenbrock,
                 _modified_schwefel, _ackley],
                [0.2, 0.2, 0.2, 0.2, 0.2], [1.0]*5
            )

        # Composition (F23–F30)
        elif fid == 23:
            self._core = self._composition(
                [_rastrigin]*5,
                sigmas=[10, 20, 30, 40, 50],
                lambdas=[1, 10, 1, 10, 1/12],
                biases=[0, 100, 200, 300, 400]
            )
        elif fid == 24:
            self._core = self._composition(
                [_ackley, _elliptic, _griewank, _rastrigin, _modified_schwefel],
                sigmas=[10, 20, 30, 40, 50],
                lambdas=[10, 1/5, 2, 1, 2],
                biases=[0, 100, 200, 300, 400]
            )
        elif fid == 25:
            self._core = self._composition(
                [_rastrigin, _happy_cat, _ackley, _discus, _rosenbrock],
                sigmas=[10, 20, 30, 40, 50],
                lambdas=[10, 1, 10, 1e-6, 1],
                biases=[0, 100, 200, 300, 400]
            )
        elif fid == 26:
            self._core = self._composition(
                [_expanded_scaffer, _modified_schwefel, _griewank,
                 _rosenbrock, _rastrigin],
                sigmas=[10, 20, 20, 30, 40],
                lambdas=[1/5, 1, 10, 1, 10],
                biases=[0, 100, 200, 300, 400]
            )
        elif fid == 27:
            self._core = self._composition(
                [_hgbat, _rastrigin, _modified_schwefel, _bent_cigar, _elliptic],
                sigmas=[10, 20, 30, 40, 50],
                lambdas=[10, 10, 2.5, 1e-26, 1e-6],
                biases=[0, 100, 200, 300, 400]
            )
        elif fid == 28:
            self._core = self._composition(
                [_ackley, _griewank, _discus, _rosenbrock, _happy_cat],
                sigmas=[10, 20, 30, 40, 50],
                lambdas=[10, 10, 1e-6, 1, 5],
                biases=[0, 100, 200, 300, 400]
            )
        elif fid == 29:
            self._core = self._composition(
                [_rastrigin, _happy_cat, _ackley, _discus, _rosenbrock,
                 _expanded_griewank_rosenbrock, _modified_schwefel, _griewank,
                 _weierstrass, _expanded_scaffer],
                sigmas=[10]*10,
                lambdas=[10, 5, 2, 1, 2, 5, 10, 5, 2, 10],
                biases=[i*100 for i in range(10)]
            )
        elif fid == 30:
            self._core = self._composition(
                [_rastrigin, _modified_schwefel, _griewank, _rosenbrock, _ackley,
                 _modified_schwefel, _expanded_scaffer, _griewank,
                 _hgbat, _rosenbrock],
                sigmas=[10]*10,
                lambdas=[10, 5, 2, 1, 2, 5, 10, 5, 2, 10],
                biases=[i*100 for i in range(10)]
            )
        else:
            raise ValueError(f"F{fid} not implemented.")

    # ------------------------------------------------------------------
    # Hybrid builders — partition z by ratios
    # ------------------------------------------------------------------

    def _make_partitions(self, z, ratios):
        dim = len(z)
        sizes = [max(1, round(r * dim)) for r in ratios]
        # Fix rounding so they sum to dim
        diff = dim - sum(sizes)
        sizes[-1] += diff
        parts = []
        idx = 0
        for s in sizes:
            parts.append(z[idx:idx+s])
            idx += s
        return parts

    def _hybrid_n(self, funcs, ratios, lambdas):
        def core(z):
            parts = self._make_partitions(z, ratios)
            return sum(lam * f(p) for f, p, lam in zip(funcs, parts, lambdas))
        return core

    def _hybrid_2(self, funcs, ratios, lambdas):
        return self._hybrid_n(funcs, ratios, lambdas)

    def _hybrid_3(self, funcs, ratios, lambdas):
        return self._hybrid_n(funcs, ratios, lambdas)

    def _hybrid_4(self, funcs, ratios, lambdas):
        return self._hybrid_n(funcs, ratios, lambdas)

    def _hybrid_5(self, funcs, ratios, lambdas):
        return self._hybrid_n(funcs, ratios, lambdas)

    # ------------------------------------------------------------------
    # Composition builder
    # ------------------------------------------------------------------

    def _composition(self, funcs, sigmas, lambdas, biases):
        n = len(funcs)
        dim = self.dim
        # Each component has its own shift/rotation
        rng = np.random.default_rng(self.func_id * 1000)
        shifts  = [rng.uniform(-80, 80, dim) for _ in range(n)]
        rotations = [_make_rotation(dim, seed=self.func_id * 1000 + i) for i in range(n)]

        def core(z_ignored):
            # z_ignored is the already globally transformed z;
            # for composition we re-apply per-component transforms from raw x.
            # We recover x = z_ignored + self.o (undo global shift)
            x = z_ignored + self.o
            z_list = [rotations[i] @ (x - shifts[i]) for i in range(n)]
            return _composition_weights(z_list, sigmas, lambdas, funcs, biases)

        return core

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, x):
        z = self._transform(x)
        return self._core(z) + self.bias

    def __call__(self, x):
        return self.evaluate(x)

    def __repr__(self):
        return (f"CEC2014 F{self.func_id} | dim={self.dim} | "
                f"bias={self.bias} | shift={'custom' if np.any(self.o) else 'none'} | "
                f"rotate={'yes' if not np.array_equal(self.M, np.eye(self.dim)) else 'no'}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_function(func_id, dim=10, shift=False, rotate=False, seed=None):
    """
    Returns a callable CEC2014Function.

    Example
    -------
        f5 = get_function(5, dim=10, shift=True, rotate=True, seed=0)
        loss = f5(some_weight_vector)
    """
    return CEC2014Function(func_id, dim=dim, shift=shift, rotate=rotate, seed=seed)


# ---------------------------------------------------------------------------
# Quick benchmark runner — drop-in for your hybrid solver
# ---------------------------------------------------------------------------

def run_benchmark(solver_factory, func_ids=range(1, 31), dim=10,
                  shift=True, rotate=True, seed=42, verbose=True):
    """
    Runs a solver against a set of CEC2014 functions.

    Parameters
    ----------
    solver_factory : callable
        Called as solver_factory(obj_func, dim) and must return an object
        with a .solve() method returning (best_sol, best_fit, curve).
    func_ids  : iterable of ints
    dim       : problem dimension
    shift, rotate, seed : passed to get_function()
    verbose   : print results

    Returns
    -------
    dict  {func_id: (best_fit, convergence_curve)}

    Example
    -------
        from cec2014 import run_benchmark
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
            print(f"  F{fid:02d}  best = {best_fit:.6f}  (bias = {f.bias})"
                  f"  error = {best_fit - f.bias:.6f}")
    return results


# ---------------------------------------------------------------------------
# Metadata table (useful for labelling plots)
# ---------------------------------------------------------------------------

FUNCTION_INFO = {
    1:  ("F1",  "Rotated High Conditioned Elliptic",          "Unimodal"),
    2:  ("F2",  "Rotated Bent Cigar",                         "Unimodal"),
    3:  ("F3",  "Rotated Discus",                             "Unimodal"),
    4:  ("F4",  "Shifted and Rotated Rosenbrock's",           "Unimodal"),
    5:  ("F5",  "Shifted and Rotated Ackley's",               "Multimodal"),
    6:  ("F6",  "Shifted and Rotated Weierstrass",            "Multimodal"),
    7:  ("F7",  "Shifted and Rotated Griewank's",             "Multimodal"),
    8:  ("F8",  "Shifted Rastrigin's",                        "Multimodal"),
    9:  ("F9",  "Rotated Rastrigin's",                        "Multimodal"),
    10: ("F10", "Shifted Schwefel's",                         "Multimodal"),
    11: ("F11", "Rotated Schwefel's",                         "Multimodal"),
    12: ("F12", "Shifted and Rotated Katsuura",               "Multimodal"),
    13: ("F13", "Shifted and Rotated HappyCat",               "Multimodal"),
    14: ("F14", "Shifted and Rotated HGBat",                  "Multimodal"),
    15: ("F15", "Shifted and Rotated Expanded Griewank+Rosen","Multimodal"),
    16: ("F16", "Shifted and Rotated Expanded Scaffer's F6",  "Multimodal"),
    17: ("F17", "Hybrid Function 1 (N=2)",                    "Hybrid"),
    18: ("F18", "Hybrid Function 2 (N=3)",                    "Hybrid"),
    19: ("F19", "Hybrid Function 3 (N=4)",                    "Hybrid"),
    20: ("F20", "Hybrid Function 4 (N=4)",                    "Hybrid"),
    21: ("F21", "Hybrid Function 5 (N=5)",                    "Hybrid"),
    22: ("F22", "Hybrid Function 6 (N=5)",                    "Hybrid"),
    23: ("F23", "Composition Function 1 (N=5)",               "Composition"),
    24: ("F24", "Composition Function 2 (N=5)",               "Composition"),
    25: ("F25", "Composition Function 3 (N=5)",               "Composition"),
    26: ("F26", "Composition Function 4 (N=5)",               "Composition"),
    27: ("F27", "Composition Function 5 (N=5)",               "Composition"),
    28: ("F28", "Composition Function 6 (N=5)",               "Composition"),
    29: ("F29", "Composition Function 7 (N=10)",              "Composition"),
    30: ("F30", "Composition Function 8 (N=10)",              "Composition"),
}