import numpy as np

from nn_core import forward_pass, activation_derivative


def backward_pass(target, outputs, weights, func):
    """Compute layer-wise delta terms for backpropagation."""
    n_layers = len(weights)
    deltas = [None] * n_layers

    output_error = target - outputs[-1]
    deltas[-1] = output_error * activation_derivative(outputs[-1], func)

    for layer_idx in range(n_layers - 2, -1, -1):
        propagated_error = deltas[layer_idx + 1] @ weights[layer_idx + 1].T
        deltas[layer_idx] = propagated_error * activation_derivative(
            outputs[layer_idx + 1], func
        )

    return deltas


def update_weights(weights, biases, deltas, outputs, learning_rate, use_bias):
    for layer_idx in range(len(weights)):
        weights[layer_idx] += learning_rate * outputs[layer_idx].T @ deltas[layer_idx]
        if use_bias:
            biases[layer_idx] += learning_rate * deltas[layer_idx]
    return weights, biases


def training_accuracy(X_train, y_train, weights, biases, func):
    """Measure accuracy on the training set."""
    correct_predictions = 0
    for sample_idx in range(X_train.shape[0]):
        x = X_train[sample_idx].reshape(1, -1)
        _, outputs = forward_pass(x, weights, biases, func)
        predicted_class = int(np.argmax(outputs[-1]))
        true_class = int(np.argmax(y_train[sample_idx]))
        if predicted_class == true_class:
            correct_predictions += 1
    return (correct_predictions / X_train.shape[0]) * 100.0


def train(X_train, y_train, weights, biases, config):
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    func = config["activation"]
    use_bias = config["use_bias"]
    n_samples = X_train.shape[0]

    accuracy_log = []

    for _ in range(epochs):
        for sample_idx in range(n_samples):
            x = X_train[sample_idx].reshape(1, -1)
            target = y_train[sample_idx].reshape(1, -1)
            _, outputs = forward_pass(x, weights, biases, func)
            deltas = backward_pass(target, outputs, weights, func)
            weights, biases = update_weights(
                weights, biases, deltas, outputs, learning_rate, use_bias
            )

        epoch_accuracy = training_accuracy(X_train, y_train, weights, biases, func)
        accuracy_log.append(epoch_accuracy)

    return weights, biases, accuracy_log
