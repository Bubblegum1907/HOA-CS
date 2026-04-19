import numpy as np
from scipy.special import gamma

def levy_step(dim, beta=1.5):
    """Computes the step size using Mantegna's algorithm."""
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
    
    sigma = (numerator / denominator)**(1 / beta)
    
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    
    step = u / (np.abs(v)**(1.0 / beta))
    return step

def apply_levy_flight(current_pos, best_pos, t, max_iter):
    dim = len(current_pos)
    
    alpha_max = 0.001 
    alpha_min = 0.00001
    alpha = alpha_max * (1 - (t / max_iter)) # Linear decay is more stable here
    alpha = max(alpha, alpha_min)
    
    step = levy_step(dim)
        
    tamed_distance = 0.01 * (current_pos - best_pos)
    
    new_pos = current_pos + alpha * step * tamed_distance
    
    noise = 0.0001 * np.random.randn(dim)
    
    return new_pos + noise