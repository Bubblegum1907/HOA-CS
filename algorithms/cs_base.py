import numpy as np
from math import gamma


def levy_flight(dim, beta=1.5):
    """Raw Lévy step vector — scaling handled by caller."""
    num   = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den   = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1,     dim)
    return u / (np.abs(v) ** (1 / beta))


def cuckoo_search(obj_func, dim, pop_size=25, max_iter=100, lb=-5, ub=5, pa=0.25):
    lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
    bound_range = ub - lb

    nests   = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(n) for n in nests])

    best_idx = np.argmin(fitness)
    best_sol = nests[best_idx].copy()
    best_fit = fitness[best_idx]
    convergence_curve = []

    for t in range(max_iter):
        # Lévy flight on one random nest
        i     = np.random.randint(0, pop_size)
        alpha = bound_range * (0.01 - 0.009 * t / max_iter)
        new_nest = nests[i] + alpha * levy_flight(dim) * (nests[i] - best_sol)
        new_nest = np.clip(new_nest, lb, ub)

        j     = np.random.randint(0, pop_size)
        f_new = obj_func(new_nest)
        if f_new < fitness[j]:
            nests[j], fitness[j] = new_nest, f_new

        # Abandon fraction pa of nests
        mask      = np.random.rand(pop_size) < pa
        if mask.any():
            r1, r2    = np.random.randint(0, pop_size, pop_size), \
                        np.random.randint(0, pop_size, pop_size)
            steps     = np.random.rand(pop_size, 1) * (nests[r1] - nests[r2])
            candidates = np.clip(nests + steps, lb, ub)
            new_fit   = np.array([obj_func(candidates[n]) for n in range(pop_size) if mask[n]])
            idx       = np.where(mask)[0]
            for k, ni in enumerate(idx):
                nests[ni]   = candidates[ni]
                fitness[ni] = new_fit[k]

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fit:
            best_fit = fitness[best_idx]
            best_sol = nests[best_idx].copy()

        convergence_curve.append(best_fit)

    return best_sol, best_fit, convergence_curve


class CuckooSearch:
    def __init__(self, obj_func, dim, pop_size=25, max_iter=100, lb=-5, ub=5, pa=0.25):
        self.obj_func = obj_func
        self.dim      = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb, self.ub, self.pa = lb, ub, pa
        self.best_sol = None

    def solve(self):
        best_sol, best_fit, curve = cuckoo_search(
            self.obj_func, self.dim, self.pop_size,
            self.max_iter, self.lb, self.ub, self.pa)
        self.best_sol = best_sol
        return best_sol, best_fit, curve