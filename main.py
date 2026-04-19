import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from algorithms.hoa_base import HOA
from algorithms.cs_base import cuckoo_search
from algorithms.hybrid import HybridHOACS
from benchmarks.NN_trainer import SimpleNN, nn_objective_function
from utils.visualisation import plot_convergence, plot_accuracy_comparison

digits = load_digits()
X, y_raw = digits.data, digits.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

input_dim = 64 
hidden_dim = 32 
output_dim = 10 

nn_model = SimpleNN(input_dim, hidden_dim, output_dim)
dim = nn_model.total_weights 

pop_size = 120 
max_iter = 200
lb, ub = -0.1, 0.1 

def fitness_wrapper(weights):
    return nn_objective_function(weights, nn_model, X_train, y_train)


hoa = HOA(fitness_wrapper, dim, pop_size, max_iter, lb, ub)
_, _, hoa_curve = hoa.solve()

cs = cuckoo_search(fitness_wrapper, dim, pop_size, max_iter, lb, ub)
_, _, cs_curve = cs.solve()

hybrid = HybridHOACS(fitness_wrapper, dim, pop_size, max_iter, lb, ub)
best_weights, best_fit, hybrid_curve = hybrid.solve()

def get_accuracy(weights, X_data, y_data):
    preds = nn_model.forward(X_data, weights)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_data, axis=1))
    return acc * 100

print("-" * 30)
print(f"HOA Final Accuracy: {get_accuracy(hoa.best_sol, X_test, y_test):.2f}%")
print(f"CS Final Accuracy: {get_accuracy(cs.best_sol, X_test, y_test):.2f}%")
print(f"Hybrid HOA-CS Accuracy: {get_accuracy(best_weights, X_test, y_test):.2f}%")
print("-" * 30)

curves = [hoa_curve, cs_curve, hybrid_curve]
labels = ['HOA (Standard)', 'Cuckoo Search (Standard)', 'Hybrid HOA-CS']

plot_convergence(curves, labels)

final_accs = [
    get_accuracy(hoa.best_sol, X_test, y_test),
    get_accuracy(cs.best_sol, X_test, y_test),
    get_accuracy(best_weights, X_test, y_test)
]
plot_accuracy_comparison(final_accs, labels)