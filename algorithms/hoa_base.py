import numpy as np


class HOA:
    """
    Hippopotamus Optimisation Algorithm — vectorised population update.
    All three phases operate on the full population matrix at once,
    replacing the per-agent Python loop with NumPy broadcasting.
    """
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.obj_func = obj_func
        self.dim      = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)

        self.X       = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.fitness = np.array([obj_func(x) for x in self.X])

        self.best_sol = self.X[np.argmin(self.fitness)].copy()
        self.best_fit = np.min(self.fitness)
        self.convergence = []

    def _eval_population(self, X_new):
        """Evaluate all candidates, keep if better than current."""
        new_fitness = np.array([self.obj_func(x) for x in X_new])
        improved    = new_fitness < self.fitness
        self.X[improved]       = X_new[improved]
        self.fitness[improved] = new_fitness[improved]

    def solve(self):
        for t in range(self.max_iter):
            # --- Phase 1: River movement (vectorised) ---
            I     = np.random.randint(1, 3, (self.pop_size, 1))
            r     = np.random.rand(self.pop_size, 1)
            X_new = self.X + r * (self.best_sol - I * self.X)
            X_new = np.clip(X_new, self.lb, self.ub)
            self._eval_population(X_new)

            # --- Phase 2: Predator defence (vectorised) ---
            radius = 2 * (1 - t / self.max_iter)
            X_new  = self.X + np.random.uniform(-1, 1, (self.pop_size, self.dim)) * radius
            X_new  = np.clip(X_new, self.lb, self.ub)
            self._eval_population(X_new)

            # --- Phase 3: Foraging (vectorised) ---
            r_idx = np.random.randint(0, self.pop_size, self.pop_size)
            r     = np.random.rand(self.pop_size, 1)
            X_new = self.X + r * (self.X[r_idx] - self.X)
            X_new = np.clip(X_new, self.lb, self.ub)
            self._eval_population(X_new)

            # --- Global best update ---
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fit:
                self.best_fit = self.fitness[best_idx]
                self.best_sol = self.X[best_idx].copy()

            self.convergence.append(self.best_fit)

        return self.best_sol, self.best_fit, self.convergence