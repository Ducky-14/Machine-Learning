import numpy as np
from activation_functions import SoftmaxActivation

class Loss:
    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def regularization_loss(self, layer):
        regularization_loss = 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

# -----------------------------------------

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

    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y.shape) == 1:
            y = np.eye(labels)[y]
        self.dinputs = -y / dvalues
        self.dinputs /= samples

# -----------------------------------------

# Requires both SoftmaxActivation and CategoricalCrossentropyLoss
class SoftmaxCrossentropyLoss(Loss):
    def __init__(self):
        self.activation_function = SoftmaxActivation()
        self.loss_function = CategoricalCrossentropyLoss()

    def forward(self, inputs, y):
        self.activation_function.forward(inputs)
        self.outputs = self.activation_function.outputs
        data_loss = self.loss_function.calculate(self.outputs, y)
        return data_loss
    
    def backward(self, y):
        self.dinputs = self.outputs.copy()
        samples = len(self.dinputs)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs[range(samples), y] -= 1
        self.dinputs /= samples