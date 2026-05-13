"""
Seven comparison algorithms: all post-2021, peer-reviewed, vectorised.
Interface: solver = Algo(obj_func, dim, pop_size, max_iter, lb, ub)
           best_sol, best_fit, curve = solver.solve()
"""

import numpy as np
from math import gamma as _gamma


# Shared helpers

def _setup(pop_size, dim, lb, ub, obj_func):
    lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
    X  = np.random.uniform(lb, ub, (pop_size, dim))
    F  = np.array([obj_func(x) for x in X])
    return lb, ub, X, F

def _best(F, X):
    i = np.argmin(F)
    return X[i].copy(), F[i]

def _clip(X, lb, ub):
    return np.clip(X, lb, ub)

def _levy_mat(n, dim, beta=1.5):
    """n × dim matrix of Lévy steps."""
    num   = _gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den   = _gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma, (n, dim))
    v = np.random.normal(0, 1,     (n, dim))
    return u / (np.abs(v) ** (1 / beta))

def _update(X, F, X_new, obj_func):
    """Evaluate X_new, keep where improved."""
    F_new    = np.array([obj_func(x) for x in X_new])
    improved = F_new < F
    X[improved] = X_new[improved]
    F[improved] = F_new[improved]


# 1. Dandelion Optimiser (DO)

class DandelionOptimiser:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.f, self.dim, self.n, self.T = obj_func, dim, pop_size, max_iter
        self.lb, self.ub, self.X, self.F = _setup(pop_size, dim, lb, ub, obj_func)
        self.best_sol, self.best_fit     = _best(self.F, self.X)

    def solve(self):
        curve = []
        for t in range(self.T):
            alpha = 1 - t / self.T
            r     = np.random.rand(self.n)
            X_new = self.X.copy()

            m1 = r < 1/3          # rising — Lévy
            m2 = (r >= 1/3) & (r < 2/3)  # descending — Brownian
            m3 = r >= 2/3          # landing

            if m1.any():
                L = _levy_mat(m1.sum(), self.dim)
                X_new[m1] = self.X[m1] + L * (self.best_sol - self.X[m1])

            if m2.any():
                X_new[m2] = self.X[m2] + \
                            np.random.randn(m2.sum(), self.dim) * alpha

            if m3.any():
                noise     = np.random.randn(m3.sum(), self.dim)
                X_new[m3] = self.best_sol + noise * \
                            (1 - t / self.T) * (self.ub - self.lb) * 0.1

            X_new = _clip(X_new, self.lb, self.ub)
            _update(self.X, self.F, X_new, self.f)
            self.best_sol, self.best_fit = _best(self.F, self.X)
            curve.append(self.best_fit)

        return self.best_sol, self.best_fit, curve


# 2. Equilibrium Optimizer (EO)

class EquilibriumOptimizer:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.f, self.dim, self.n, self.T = obj_func, dim, pop_size, max_iter
        self.lb, self.ub, self.X, self.F = _setup(pop_size, dim, lb, ub, obj_func)
        self.best_sol, self.best_fit     = _best(self.F, self.X)

    def solve(self):
        curve = []
        a1, a2 = 2.0, 1.0
        for t in range(self.T):
            si      = np.argsort(self.F)
            pool    = list(self.X[si[:4]]) + [np.mean(self.X[si[:4]], axis=0)]
            t_val   = (1 - t / self.T) ** (a2 * self.T)

            Ceq_idx = np.random.randint(0, 5, self.n)
            Ceq     = np.array([pool[k] for k in Ceq_idx])
            lam     = np.random.uniform(0, 1, (self.n, self.dim))
            r       = np.random.uniform(0, 1, (self.n, self.dim))
            F_      = a1 * np.sign(r - 0.5) * (np.exp(-lam * t_val) - 1)
            GCP     = np.where(np.random.rand(self.n, 1) > 0.5,
                               0.5 * np.random.rand(self.n, 1), 0)
            G0      = GCP * (Ceq - lam * self.X)
            X_new   = Ceq + (self.X - Ceq) * F_ + (G0 / (lam + 1e-10)) * (1 - F_)
            X_new   = _clip(X_new, self.lb, self.ub)
            _update(self.X, self.F, X_new, self.f)
            self.best_sol, self.best_fit = _best(self.F, self.X)
            curve.append(self.best_fit)

        return self.best_sol, self.best_fit, curve


# 3. RIME

class RIME:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.f, self.dim, self.n, self.T = obj_func, dim, pop_size, max_iter
        self.lb, self.ub, self.X, self.F = _setup(pop_size, dim, lb, ub, obj_func)
        self.best_sol, self.best_fit     = _best(self.F, self.X)

    def solve(self):
        curve = []
        w = 5.0
        for t in range(self.T):
            E     = np.sqrt(t / self.T)
            X_new = self.X.copy()

            # Soft random dims move toward best
            mask  = np.random.rand(self.n, self.dim) < E
            noise = w * np.random.randn(self.n, self.dim) * (1 - t / self.T)
            X_new[mask] = (np.broadcast_to(self.best_sol, (self.n, self.dim)) + noise)[mask]

            # replace dims with prob based on normalised fitness
            norm_f    = self.F / (self.best_fit + 1e-10)
            hard_prob = 1 / (1 + np.exp(-norm_f))
            hmask     = np.random.rand(self.n, self.dim) < hard_prob[:, None]
            X_new[hmask] = np.broadcast_to(self.best_sol, (self.n, self.dim))[hmask]

            X_new = _clip(X_new, self.lb, self.ub)
            _update(self.X, self.F, X_new, self.f)
            self.best_sol, self.best_fit = _best(self.F, self.X)
            curve.append(self.best_fit)

        return self.best_sol, self.best_fit, curve


# 4. Fox Optimiser (FOX) — Mohammed & Rashid, 2023

class FoxOptimiser:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.f, self.dim, self.n, self.T = obj_func, dim, pop_size, max_iter
        self.lb, self.ub, self.X, self.F = _setup(pop_size, dim, lb, ub, obj_func)
        self.best_sol, self.best_fit     = _best(self.F, self.X)

    def solve(self):
        curve = []
        for t in range(self.T):
            tt    = t / self.T
            r1    = np.random.rand(self.n, 1)
            r2    = np.random.rand(self.n, 1)
            mask  = np.random.rand(self.n) < 0.5

            X_new = self.X.copy()
            if mask.any():
                jump         = 0.5 * (self.ub - self.lb) * (1 - tt)
                signs        = np.random.choice([-1, 1], (mask.sum(), self.dim))
                X_new[mask]  = self.best_sol + r1[mask] * jump * signs

            if (~mask).any():
                rand_pos         = np.random.uniform(self.lb[0], self.ub[0],
                                                     (~mask).sum() * self.dim
                                                     ).reshape(-1, self.dim)
                mid              = 0.5 * (self.best_sol + rand_pos)
                X_new[~mask]     = mid + r2[~mask] * (mid - self.X[~mask])

            X_new = _clip(X_new, self.lb, self.ub)
            _update(self.X, self.F, X_new, self.f)
            self.best_sol, self.best_fit = _best(self.F, self.X)
            curve.append(self.best_fit)

        return self.best_sol, self.best_fit, curve


# 5. Giant Trevally Optimizer (GTO) — Sadiq et al., 2022

class GiantTrevallyOptimizer:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.f, self.dim, self.n, self.T = obj_func, dim, pop_size, max_iter
        self.lb, self.ub, self.X, self.F = _setup(pop_size, dim, lb, ub, obj_func)
        self.best_sol, self.best_fit     = _best(self.F, self.X)

    def solve(self):
        curve = []
        A = 0.1
        for t in range(self.T):
            theta = np.random.uniform(10, 90) * np.pi / 180
            r     = np.random.rand(self.n)
            X_new = self.X.copy()

            m1 = r < 1/3
            m2 = (r >= 1/3) & (r < 2/3)
            m3 = r >= 2/3

            if m1.any():
                noise       = np.random.randn(m1.sum(), self.dim)
                X_new[m1]   = self.X[m1] + A * noise * \
                              (self.best_sol - self.X[m1]) / np.cos(theta)

            if m2.any():
                target      = self.best_sol - np.tan(theta) * \
                              np.abs(self.best_sol - self.X[m2])
                X_new[m2]   = self.X[m2] + \
                              np.random.rand(m2.sum(), 1) * (target - self.X[m2])

            if m3.any():
                noise       = np.random.randn(m3.sum(), self.dim)
                X_new[m3]   = self.best_sol + noise * \
                              np.abs(self.best_sol - self.X[m3]) * (1 - t / self.T)

            X_new = _clip(X_new, self.lb, self.ub)
            _update(self.X, self.F, X_new, self.f)
            self.best_sol, self.best_fit = _best(self.F, self.X)
            curve.append(self.best_fit)

        return self.best_sol, self.best_fit, curve


# 6. Aquila Optimiser (AO)

class AquilaOptimiser:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.f, self.dim, self.n, self.T = obj_func, dim, pop_size, max_iter
        self.lb, self.ub, self.X, self.F = _setup(pop_size, dim, lb, ub, obj_func)
        self.best_sol, self.best_fit     = _best(self.F, self.X)

    def solve(self):
        curve = []
        for t in range(self.T):
            mean_X = np.mean(self.X, axis=0)
            r      = np.random.rand(self.n)
            X_new  = self.X.copy()
            L      = _levy_mat(self.n, self.dim)

            if t <= self.T * 2/3:
                m1 = r < 0.5
                if m1.any():
                    X_new[m1] = self.best_sol * (1 - t / self.T) + \
                                (mean_X - self.best_sol * np.random.rand()) * L[m1]
                if (~m1).any():
                    ri        = np.random.randint(0, self.n, (~m1).sum())
                    D         = np.abs(self.X[ri] - self.X[~m1])
                    X_new[~m1]= self.best_sol + np.random.randn((~m1).sum(), self.dim) * D
            else:
                m1 = r < 0.5
                if m1.any():
                    X_new[m1] = (self.best_sol - mean_X) * np.random.rand() - \
                                np.random.rand() + \
                                np.random.uniform(self.lb, self.ub, (m1.sum(), self.dim))
                if (~m1).any():
                    ri         = np.random.randint(0, self.n, (~m1).sum())
                    X_new[~m1] = self.best_sol * L[~m1] + \
                                 self.X[ri] * np.random.rand((~m1).sum(), 1)

            X_new = _clip(X_new, self.lb, self.ub)
            _update(self.X, self.F, X_new, self.f)
            self.best_sol, self.best_fit = _best(self.F, self.X)
            curve.append(self.best_fit)

        return self.best_sol, self.best_fit, curve


# African Vultures Optimisation

class AfricanVulturesOptimisation:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.f, self.dim, self.n, self.T = obj_func, dim, pop_size, max_iter
        self.lb, self.ub, self.X, self.F = _setup(pop_size, dim, lb, ub, obj_func)
        self.best_sol, self.best_fit     = _best(self.F, self.X)

    def solve(self):
        curve = []
        p1, p2, p3 = 0.6, 0.4, 0.6
        for t in range(self.T):
            si  = np.argsort(self.F)
            V1  = self.X[si[0]]
            V2  = self.X[si[1]]
            F_  = (2 * np.random.rand(self.n, 1) + 1) * \
                  (1 - t / self.T) + \
                  np.random.randn(self.n, 1) * 0.1

            ref = np.where(np.random.rand(self.n, 1) < p1, V1, V2)
            X_new = self.X.copy()

            expl = np.abs(F_[:, 0]) >= 1
            expt = ~expl

            if expl.any():
                X_new[expl] = ref[expl] - \
                              np.abs(ref[expl] - self.X[expl]) * F_[expl]

            if expt.any():
                sub = np.where(np.random.rand(expt.sum(), 1) < p2)
                idx = np.where(expt)[0]

                p2_mask = np.random.rand(expt.sum()) < p2
                i_p2    = idx[p2_mask]
                i_np2   = idx[~p2_mask]

                if len(i_p2):
                    d          = ref[i_p2] - self.X[i_p2]
                    X_new[i_p2]= d * (np.random.rand(len(i_p2), 1) + F_[i_p2]) + \
                                 np.random.randn(len(i_p2), self.dim) * 0.01

                if len(i_np2):
                    p3_mask    = np.random.rand(len(i_np2)) < p3
                    i_lev      = i_np2[p3_mask]
                    i_rot      = i_np2[~p3_mask]

                    if len(i_lev):
                        L          = _levy_mat(len(i_lev), self.dim)
                        X_new[i_lev] = ref[i_lev] - \
                                       np.abs(ref[i_lev] - self.X[i_lev]) * F_[i_lev] * L

                    if len(i_rot):
                        A          = V1 - V2
                        X_new[i_rot] = ref[i_rot] - A * np.random.rand(len(i_rot), 1)

            X_new = _clip(X_new, self.lb, self.ub)
            _update(self.X, self.F, X_new, self.f)
            self.best_sol, self.best_fit = _best(self.F, self.X)
            curve.append(self.best_fit)

        return self.best_sol, self.best_fit, curve


# Registry

COMPARISON_ALGORITHMS = {
    "DO"   : DandelionOptimiser,
    "EO"   : EquilibriumOptimizer,
    "RIME" : RIME,
    "FOX"  : FoxOptimiser,
    "GTO"  : GiantTrevallyOptimizer,
    "AO"   : AquilaOptimiser,
    "AVOA" : AfricanVulturesOptimisation,
}