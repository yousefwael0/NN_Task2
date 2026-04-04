import numpy as np
import csv


def load_data(filepath, num_classes=3, samples_per_class=50, train_per_class=30):
    species_to_class = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    data_by_class = {i: [] for i in range(num_classes)}

    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            species = row[0].strip()
            features = list(map(float, row[1:6]))
            label = species_to_class[species]
            data_by_class[label].append(features)

    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []

    for cls, rows in data_by_class.items():
        rows = np.array(rows)
        X_train_parts.append(rows[:train_per_class])
        y_train_parts.append([cls] * train_per_class)
        X_test_parts.append(rows[train_per_class:samples_per_class])
        y_test_parts.append([cls] * (samples_per_class - train_per_class))

    X_train = np.vstack(X_train_parts)
    X_test = np.vstack(X_test_parts)

    # Z-score normalise using train statistics only (applied to both sets)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # avoid division by zero
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    y_train = one_hot(np.concatenate(y_train_parts), num_classes)
    y_test = one_hot(np.concatenate(y_test_parts), num_classes)

    return X_train, y_train, X_test, y_test


def one_hot(labels, num_classes):
    n = len(labels)
    encoded = np.zeros((n, num_classes))
    for i, label in enumerate(labels):
        encoded[i, int(label)] = 1.0
    return encoded


def init_weights(layer_sizes, use_bias):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
        w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
        if use_bias:
            b = np.random.randn(1, layer_sizes[i + 1]) * scale
        else:
            b = np.zeros((1, layer_sizes[i + 1]))
        weights.append(w)
        biases.append(b)
    return weights, biases


def activation(z, func):
    if func == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    elif func == "tanh":
        return np.tanh(z)
    else:
        raise ValueError


def activation_derivative(a, func):
    if func == "sigmoid":
        return a * (1.0 - a)
    elif func == "tanh":
        return 1.0 - a**2
    else:
        raise ValueError


def forward_pass(x, weights, biases, func):
    x = x.reshape(1, -1)
    nets, outputs = [], [x]
    for w, b in zip(weights, biases):
        z = outputs[-1] @ w + b
        a = activation(z, func)
        nets.append(z)
        outputs.append(a)
    return nets, outputs


def build_layer_sizes(n_features, hidden_layers, n_classes):
    return [n_features] + hidden_layers + [n_classes]