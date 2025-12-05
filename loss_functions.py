import numpy as np
from activation_functions import SoftmaxActivation

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

# -----------------------------------------

# Requires both SoftmaxActivation and CategoricalCrossentropyLoss
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