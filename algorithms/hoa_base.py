import numpy as np

class HOA:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=100, lb=-100, ub=100):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
        
        # Initialize Hippos
        self.X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.array([self.obj_func(x) for x in self.X])
        
        self.best_sol = self.X[np.argmin(self.fitness)].copy()
        self.best_fit = np.min(self.fitness)
        self.convergence = []

    def solve(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # --- Phase 1: Stay in the River (Exploration) ---
                # Hippo moves toward the best solution or a random hippo
                I = np.random.randint(1, 3) 
                X_new = self.X[i] + np.random.rand() * (self.best_sol - I * self.X[i])
                X_new = np.clip(X_new, self.lb, self.ub)
                
                f_new = self.obj_func(X_new)
                if f_new < self.fitness[i]:
                    self.X[i], self.fitness[i] = X_new, f_new

                # --- Phase 2: Defense against Predators (Exploitation) ---
                # Small movements to refine the position
                radius = 2 * (1 - t / self.max_iter) # Decreasing search radius
                X_new = self.X[i] + (np.random.uniform(-1, 1, self.dim) * radius)
                X_new = np.clip(X_new, self.lb, self.ub)
                
                f_new = self.obj_func(X_new)
                if f_new < self.fitness[i]:
                    self.X[i], self.fitness[i] = X_new, f_new

                # --- Phase 3: Foraging/Escaping ---
                # Search for a new area based on a random walk
                r1 = np.random.randint(0, self.pop_size)
                X_new = self.X[i] + np.random.rand() * (self.X[r1] - self.X[i])
                X_new = np.clip(X_new, self.lb, self.ub)
                
                f_new = self.obj_func(X_new)
                if f_new < self.fitness[i]:
                    self.X[i], self.fitness[i] = X_new, f_new

            # Update Global Best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fit:
                self.best_fit = self.fitness[best_idx]
                self.best_sol = self.X[best_idx].copy()
                
            self.convergence.append(self.best_fit)
            
        return self.best_sol, self.best_fit, self.convergence