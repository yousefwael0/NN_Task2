import numpy as np

from nn_core import forward_pass, activation_derivative


def backward_pass(d, outputs, weights, func):
    n_layers = len(weights)
    deltas = [None] * n_layers
    output_error = d - outputs[-1]
    deltas[-1] = output_error * activation_derivative(outputs[-1], func)
    for l in range(n_layers - 2, -1, -1):
        error = deltas[l + 1] @ weights[l + 1].T
        deltas[l] = error * activation_derivative(outputs[l + 1], func)
    return deltas


def update_weights(weights, biases, deltas, outputs, eta, use_bias):
    for l in range(len(weights)):
        weights[l] += eta * outputs[l].T @ deltas[l]
        if use_bias:
            biases[l] += eta * deltas[l]
    return weights, biases


def train(X_train, y_train, weights, biases, config):
    eta = config["eta"]
    epochs = config["epochs"]
    func = config["activation"]
    use_bias = config["use_bias"]
    n_samples = X_train.shape[0]

    accuracy_log = []

    for epoch in range(epochs):
        for i in range(n_samples):
            x = X_train[i].reshape(1, -1)
            d = y_train[i].reshape(1, -1)
            _, outputs = forward_pass(x, weights, biases, func)
            deltas = backward_pass(d, outputs, weights, func)
            weights, biases = update_weights(
                weights, biases, deltas, outputs, eta, use_bias
            )

        correct = 0
        for i in range(n_samples):
            x = X_train[i].reshape(1, -1)
            _, outs = forward_pass(x, weights, biases, func)
            pred = np.argmax(outs[-1])
            true = np.argmax(y_train[i])
            if pred == true:
                correct += 1

        acc = correct / n_samples * 100
        accuracy_log.append(acc)

    return weights, biases, accuracy_log