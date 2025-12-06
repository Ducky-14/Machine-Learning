import numpy as np
import optuna
from nnfs.datasets import spiral_data, vertical_data

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, bias_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l2=0):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
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

class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities

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
        if not hasattr(layer, "weight_momentums"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases
        
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1


# Create Datasets
X_train, y_train = spiral_data(samples=100, classes=3)
X_validate, y_validate = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Fixed Training Parameters
TRAINING_BATCH_SIZE = 100
TRAINING_EPOCHS = 10001

# Optuna Tuning Parameters
OPTUNA_EPOCHS = 1000
OPTUNA_TRIALS = 30

DEFAULT_HYPERPARAMETERS = {
    "n_neurons": 64,
    "learning_rate": 0.01,
    "decay": 5e-5,
    "momentum": 0.99,
    "rho": 0.999,
    "weight_regularizer_l2": 1e-4,
    "bias_regularizer_l2": 1e-4,
    "dropout_probability": 0.1
}

def train_model_hyperparameters(X, y, X_validate, y_validate, hyperparameters, epochs):
    n_neurons = hyperparameters["n_neurons"]
    learning_rate = hyperparameters["learning_rate"]
    decay = hyperparameters["decay"]
    momentum = hyperparameters["momentum"]
    rho = hyperparameters["rho"]
    weight_regularizer_l2 = hyperparameters["weight_regularizer_l2"]
    bias_regularizer_l2 = hyperparameters["bias_regularizer_l2"]
    dropout_probability = hyperparameters["dropout_probability"]

    dense1 = DenseLayer(2, n_neurons, weight_regularizer_l2=weight_regularizer_l2, bias_regularizer_l2=bias_regularizer_l2)
    activation1 = ReLUActivation()
    dropout1 = DropoutLayer(dropout_probability)
    dense2 = DenseLayer(n_neurons, 3)
    loss_activation = SoftmaxCrossentropyLoss()
    optimizer = AdamOptimizer(learning_rate, decay, beta1=momentum, beta2=rho)

    dropout1.training = True

    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.outputs)
        dropout1.forward(activation1.outputs)
        dense2.forward(dropout1.outputs)
        loss_activation.forward(dense2.outputs, y)

        loss_activation.backward(y)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    dropout1.training = False
    dense1.forward(X_validate)
    activation1.forward(dense1.outputs)
    dropout1.forward(activation1.outputs)
    dense2.forward(dropout1.outputs)
    validate_accuracy = check_accuracy(dense2.outputs, y_validate)
    return validate_accuracy

def objective(trial):
    hyperparameters = {
        "n_neurons": trial.suggest_int("n_neurons", 16, 256),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "decay": trial.suggest_loguniform("decay", 1e-6, 1e-2),
        "momentum": trial.suggest_float("momentum", 0, 0.999),
        "rho": trial.suggest_float("rho", 0, 0.9999),
        "weight_regularizer_l2": trial.suggest_loguniform("weight_regularizer_l2", 1e-6, 1e-2),
        "bias_regularizer_l2": trial.suggest_loguniform("bias_regularizer_l2", 1e-6, 1e-2),
        "dropout_probability": trial.suggest_float("dropout_probability", 0, 0.5)
    }
    validate_accuracy = train_model_hyperparameters(X_train, y_train, X_validate, y_validate, hyperparameters, OPTUNA_EPOCHS)
    return validate_accuracy

# Hyperparameters Tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=OPTUNA_TRIALS)
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation accuracy: {study.best_value}")
DEFAULT_HYPERPARAMETERS.update(study.best_params)

# Unpack Hyperparameters
n_neurons = DEFAULT_HYPERPARAMETERS["n_neurons"]
learning_rate = DEFAULT_HYPERPARAMETERS["learning_rate"]
decay = DEFAULT_HYPERPARAMETERS["decay"]
momentum = DEFAULT_HYPERPARAMETERS["momentum"]
rho = DEFAULT_HYPERPARAMETERS["rho"]
weight_regularizer_l2 = DEFAULT_HYPERPARAMETERS["weight_regularizer_l2"]
bias_regularizer_l2 = DEFAULT_HYPERPARAMETERS["bias_regularizer_l2"]
dropout_probability = DEFAULT_HYPERPARAMETERS["dropout_probability"]

# Build Model
dense1 = DenseLayer(2, n_neurons, weight_regularizer_l2=weight_regularizer_l2, bias_regularizer_l2=bias_regularizer_l2)
activation1 = ReLUActivation()
dropout1 = DropoutLayer(dropout_probability)
dense2 = DenseLayer(n_neurons, 3)
loss_activation = SoftmaxCrossentropyLoss()
optimizer = AdamOptimizer(learning_rate, decay, beta1=momentum, beta2=rho)

# Training
def train_model(X, y, epochs, training=False, testing=False):
    if training:
        dropout1.training = True
    
    # Training Loop
    for epoch in range(epochs):
        # Forward Pass
        dense1.forward(X)
        activation1.forward(dense1.outputs)
        dropout1.forward(activation1.outputs)
        dense2.forward(dropout1.outputs)
        data_loss = loss_activation.forward(dense2.outputs, y)
        regularization_loss = loss_activation.regularization_loss(dense1) + loss_activation.regularization_loss(dense2)
        loss = data_loss + regularization_loss
        accuracy = check_accuracy(dense2.outputs, y)

        if testing:
            return accuracy, loss

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} \t" +
                f"Accuracy: {accuracy: .3f} \t" +
                f"Loss: {loss: .3f} " +
                f"(Data: {data_loss: .3f}  +  " +
                f"Reg: {regularization_loss: .3f}) \t" +
                f"lr: {optimizer.current_learning_rate: .5f}")

        # Backward Pass
        loss_activation.backward(y)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        # Update Weights and Biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

print("Training")
train_model(X_train, y_train, TRAINING_EPOCHS, training=True)

# Testing
testing_accuracy, testing_loss = train_model(X_test, y_test, 1, testing=True)
print("\033[31mTesting \t\033[0m" +
      f"Accuracy: {testing_accuracy: .3f} \t" +
      f"Loss: {testing_loss: .3f}")
