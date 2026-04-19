import numpy as np
from utils.levy_flight import apply_levy_flight

class HybridHOACS:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-0.1, ub=0.1):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
        
        self.X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.array([self.obj_func(ind) for ind in self.X])
        
        self.best_sol = self.X[np.argmin(self.fitness)].copy()
        self.best_fit = np.min(self.fitness)
        
        self.stagnation_counter = 0
        self.convergence_curve = []

    def solve(self):
        for t in range(self.max_iter):
            sorted_indices = np.argsort(self.fitness)
            best_indices = sorted_indices[:int(self.pop_size * 0.7)]
            
            for i in range(self.pop_size):
                if i in best_indices:
                    I = np.random.randint(1, 3)
                    X_new = self.X[i] + np.random.rand() * (self.best_sol - I * self.X[i])
                else:
                    X_new = apply_levy_flight(self.X[i], self.best_sol, t, self.max_iter)

                X_new = np.clip(X_new, self.lb, self.ub)
                f_new = self.obj_func(X_new)
                if f_new < self.fitness[i]:
                    self.X[i], self.fitness[i] = X_new, f_new

            if self.stagnation_counter > 5:
                for i in range(self.pop_size):
                    predator = np.random.uniform(self.lb, self.ub)
                    X_new = self.X[i] + np.random.rand() * (self.X[i] - predator)
                    
                    X_new = np.clip(X_new, self.lb, self.ub)
                    f_new = self.obj_func(X_new)
                    if f_new < self.fitness[i]:
                        self.X[i], self.fitness[i] = X_new, f_new

            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fit:
                self.best_fit = self.fitness[current_best_idx]
                self.best_sol = self.X[current_best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            self.convergence_curve.append(self.best_fit)
            
        return self.best_sol, self.best_fit, self.convergence_curve