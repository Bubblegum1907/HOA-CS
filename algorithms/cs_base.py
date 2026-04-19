import numpy as np

def cuckoo_search(obj_func, dim, pop_size=25, max_iter=100, lb=-5, ub=5, pa=0.25):
    # 1. Initialization
    lower_boundary = np.array(lb)
    upper_boundary = np.array(ub)
    
    # Nests represent the population of potential solutions
    nests = np.random.uniform(lower_boundary, upper_boundary, (pop_size, dim))
    fitness = np.array([obj_func(n) for n in nests])
    
    # Find the initial best
    best_idx = np.argmin(fitness)
    best_sol = nests[best_idx].copy()
    best_fit = fitness[best_idx]
    
    convergence_curve = []

    for t in range(max_iter):
        # --- Step A: Generate a new cuckoo via Lévy Flights ---
        # In classic CS, we pick a random nest and move it
        i = np.random.randint(0, pop_size)
        new_nest = levy_flight(nests[i], best_sol, dim)
        new_nest = np.clip(new_nest, lower_boundary, upper_boundary)
        
        # Evaluate and pick a random nest 'j' to potentially replace
        j = np.random.randint(0, pop_size)
        f_new = obj_func(new_nest)
        
        if f_new < fitness[j]:
            nests[j] = new_nest
            fitness[j] = f_new

        # --- Step B: Discovery of alien eggs (Abandoning Nests) ---
        # Host bird discovers cuckoo eggs with probability Pa
        for n in range(pop_size):
            if np.random.rand() < pa:
                # Random walk using two random nests
                r1, r2 = np.random.randint(0, pop_size, 2)
                step_size = np.random.rand() * (nests[r1] - nests[r2])
                nests[n] = nests[n] + step_size
                nests[n] = np.clip(nests[n], lower_boundary, upper_boundary)
                fitness[n] = obj_func(nests[n])

        # Update the global best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fit:
            best_fit = fitness[current_best_idx]
            best_sol = nests[current_best_idx].copy()
            
        convergence_curve.append(best_fit)

    return best_sol, best_fit, convergence_curve

def levy_flight(current_nest, best_sol, dim):
    """
    Standard Mantegna's algorithm for Lévy Flights.
    """
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
            (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v)**(1 / beta))
    
    # The step is scaled relative to the distance from the best solution
    step_size = 0.01 * step * (current_nest - best_sol)
    new_nest = current_nest + step_size * np.random.normal(dim)
    return new_nest