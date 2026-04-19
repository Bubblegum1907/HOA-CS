import numpy as np

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.total_weights = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim) + output_dim

    def decode_weights(self, weights_vector):
        """Splits a single flat vector into weight matrices and biases."""
        idx = 0
        
        w1_end = idx + (self.input_dim * self.hidden_dim)
        w1 = weights_vector[idx:w1_end].reshape(self.input_dim, self.hidden_dim)
        idx = w1_end
        
        b1_end = idx + self.hidden_dim
        b1 = weights_vector[idx:b1_end]
        idx = b1_end
        
        w2_end = idx + (self.hidden_dim * self.output_dim)
        w2 = weights_vector[idx:w2_end].reshape(self.hidden_dim, self.output_dim)
        idx = w2_end
        
        b2 = weights_vector[idx:]
        
        return w1, b1, w2, b2

    def forward(self, X, weights_vector):
        w1, b1, w2, b2 = self.decode_weights(weights_vector)
        
        z1 = np.dot(X, w1) + b1
        a1 = np.tanh(z1) 
        
        z2 = np.dot(a1, w2) + b2
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True)) 
        softmax_out = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        
        return softmax_out

def nn_objective_function(weights_vector, model, X, y):
    """
    The fitness function. 
    Lowering the L2 penalty slightly helps with MNIST exploration.
    """
    l2_penalty = 0.001 * np.sum(weights_vector**2)
    predictions = model.forward(X, weights_vector)
    
    m = y.shape[0]
    loss = -np.sum(y * np.log(predictions + 1e-15)) / m
    return loss + l2_penalty