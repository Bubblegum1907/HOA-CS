import numpy as np
from algorithms.cs_base import levy_flight


class HybridHOACS:
    """
    Hybrid HOA + Cuckoo Search.
    Early iterations: best 40% explore via Levy, rest do HOA river move.
    Late iterations:  best 40% exploit via HOA local search, rest forage.
    Stagnation: scatter worst 20% only.
    """
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100,
                 lb=-0.1, ub=0.1, stagnation_limit=10, levy_frac=0.5):
        self.obj_func         = obj_func
        self.dim              = dim
        self.pop_size         = pop_size
        self.max_iter         = max_iter
        self.stagnation_limit = stagnation_limit
        self.levy_frac        = levy_frac

        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
        self.bound_range = self.ub - self.lb

        self.X       = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.fitness = np.array([obj_func(x) for x in self.X])

        self.best_sol           = self.X[np.argmin(self.fitness)].copy()
        self.best_fit           = np.min(self.fitness)
        self.stagnation_counter = 0
        self.convergence_curve  = []

    def _eval_candidates(self, X_new, indices):
        """Evaluate X_new[indices], update in place if improved."""
        for i in indices:
            f = self.obj_func(X_new[i])
            if f < self.fitness[i]:
                self.X[i], self.fitness[i] = X_new[i], f

    def solve(self):
        exploration_cutoff = int(self.max_iter * self.levy_frac)
        top_n    = int(self.pop_size * 0.4)
        bottom_n = int(self.pop_size * 0.2)

        for t in range(self.max_iter):
            sorted_idx    = np.argsort(self.fitness)
            best_idx      = sorted_idx[:top_n]
            rest_idx      = sorted_idx[top_n:]
            worst_idx     = sorted_idx[-bottom_n:]
            in_exploration = t < exploration_cutoff

            X_new = self.X.copy()

            if in_exploration:
                # Best agents - Levy flight
                alpha = self.bound_range * (0.01 - 0.009 * t / self.max_iter)
                for i in best_idx:
                    step     = levy_flight(self.dim)
                    X_new[i] = self.X[i] + alpha * step * (self.X[i] - self.best_sol)

                # Rest - HOA river move
                I = np.random.randint(1, 3, (len(rest_idx), 1))
                r = np.random.rand(len(rest_idx), 1)
                X_new[rest_idx] = self.X[rest_idx] + \
                                  r * (self.best_sol - I * self.X[rest_idx])
            else:
                # Best agents - HOA shrinking local search
                radius = 2 * (1 - t / self.max_iter)
                X_new[best_idx] = self.X[best_idx] + \
                    np.random.uniform(-1, 1, (top_n, self.dim)) * radius

                # Rest - HOA foraging
                r_idx = np.random.randint(0, self.pop_size, len(rest_idx))
                r     = np.random.rand(len(rest_idx), 1)
                X_new[rest_idx] = self.X[rest_idx] + \
                                  r * (self.X[r_idx] - self.X[rest_idx])

            X_new = np.clip(X_new, self.lb, self.ub)
            self._eval_candidates(X_new, range(self.pop_size))

            # Stagnation scatter - worst 20% only
            if self.stagnation_counter >= self.stagnation_limit:
                for i in worst_idx:
                    self.X[i]       = np.random.uniform(self.lb, self.ub)
                    self.fitness[i] = self.obj_func(self.X[i])
                self.stagnation_counter = 0

            best_i = np.argmin(self.fitness)
            if self.fitness[best_i] < self.best_fit:
                self.best_fit           = self.fitness[best_i]
                self.best_sol           = self.X[best_i].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            self.convergence_curve.append(self.best_fit)

        return self.best_sol, self.best_fit, self.convergence_curve