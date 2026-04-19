import numpy as np

def sphere(x):
    """
    Continuous, Convex, Unimodal.
    Global Minimum: 0 at [0, 0, ..., 0]
    """
    return np.sum(x**2)

def rastrigin(x):
    """
    Highly Multimodal (lots of local minima).
    Global Minimum: 0 at [0, 0, ..., 0]
    """
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    """
    Global Minimum: 0 at [0, 0, ..., 0]
    """
    dim = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / dim)
    return term1 + term2 + 20 + np.e

def rosenbrock(x):
    """
    The 'Banana Function'. The minimum is in a long, narrow, flat valley.
    Global Minimum: 0 at [1, 1, ..., 1]
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)