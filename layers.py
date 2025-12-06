import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, bias_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
            
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        self.dinputs = np.dot(dvalues, self.weights.T)

# -----------------------------------------

class DropoutLayer:
    def __init__(self, dropout_prob):
        self.keep_prob = 1 - dropout_prob
        self.training = True
    
    def forward(self, inputs):
        if self.training:
            self.inputs = inputs
            self.binary_mask = np.random.binomial(1, self.keep_prob, size=inputs.shape) / self.keep_prob
            self.outputs = inputs * self.binary_mask
        else:
            self.outputs = inputs
    
    def backward(self, dvalues):
        if self.training:
            self.dinputs = dvalues * self.binary_mask
        else:
            self.dinputs = dvalues
