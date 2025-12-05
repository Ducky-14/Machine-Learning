import numpy as np
from nnfs.datasets import spiral_data, vertical_data

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossentropyLoss(Loss):
    def forward(self, y_predictions, y):
        samples = len(y_predictions)
        y_predictions_clipped = np.clip(y_predictions, 1e-7, 1-1e-7)

        if len(y.shape) == 1:
            correct_confidences = y_predictions_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_predictions_clipped * y, axis=1)
        
        log_likelihoods = -np.log(correct_confidences)
        return log_likelihoods

class SoftmaxCrossentropyLoss:
    def __init__(self):
        self.activation_function = SoftmaxActivation()
        self.loss_function = CategoricalCrossentropyLoss()

    def forward(self, inputs, y):
        self.activation_function.forward(inputs)
        self.output = self.activation_function.output
        self.loss = self.loss_function.calculate(self.output, y)
        self.accuracy = check_accuracy(self.output, y)
    
    def backward(self, y):
        self.dinputs = self.output.copy()
        samples = len(self.dinputs)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs[range(samples), y] -= 1
        self.dinputs /= samples

def check_accuracy(predictions, y):
    predictions = np.argmax(predictions, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, decay=0, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))
        
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    
    def post_update_params(self):
        self.iterations += 1


# Create Dataset
X, y = spiral_data(samples=100, classes=3)

# Build Model
batch_size = 100
epochs = 10001

learning_rate = 0.02
decay = 1e-5
rho = 0.999
momentum = 0.9

dense1 = LayerDense(2, 64)
activation1 = ReLUActivation()
dense2 = LayerDense(64, 3)
loss_activation = SoftmaxCrossentropyLoss()
optimizer = AdamOptimizer(learning_rate, decay, momentum, rho)

def forward_pass():
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_activation.forward(dense2.output, y)

def backward_pass():
    loss_activation.backward(y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

for epoch in range(epochs):
    forward_pass()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} \t" + 
              f"Loss: {loss_activation.loss: .3f} \t" + 
              f"Accuracy: {loss_activation.accuracy: .3f} \t" + 
              f"lr: {optimizer.current_learning_rate: .3f}")
        
    backward_pass()

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
