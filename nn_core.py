import csv

import numpy as np

SPECIES_TO_CLASS = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}


def load_data(filepath, num_classes=3, samples_per_class=50, train_per_class=30):
    """Load penguin data, split by class, then normalize from train statistics."""
    samples_by_class = {class_id: [] for class_id in range(num_classes)}

    with open(filepath, newline="", encoding="utf-8") as dataset_file:
        reader = csv.reader(dataset_file)
        next(reader)  # skip header
        for row in reader:
            species = row[0].strip()
            features = [float(value) for value in row[1:6]]
            class_id = SPECIES_TO_CLASS[species]
            samples_by_class[class_id].append(features)

    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []

    for class_id, samples in samples_by_class.items():
        samples = np.array(samples)
        X_train_parts.append(samples[:train_per_class])
        y_train_parts.append([class_id] * train_per_class)
        X_test_parts.append(samples[train_per_class:samples_per_class])
        y_test_parts.append([class_id] * (samples_per_class - train_per_class))

    X_train = np.vstack(X_train_parts)
    X_test = np.vstack(X_test_parts)

    # Keep normalization consistent between train/test by using train-only stats.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    y_train = one_hot(np.concatenate(y_train_parts), num_classes)
    y_test = one_hot(np.concatenate(y_test_parts), num_classes)

    return X_train, y_train, X_test, y_test


def one_hot(labels, num_classes):
    """Convert integer labels into one-hot encoded vectors."""
    encoded = np.zeros((len(labels), num_classes))
    for row_idx, label in enumerate(labels):
        encoded[row_idx, int(label)] = 1.0
    return encoded


def init_weights(layer_sizes, use_bias):
    """Initialize layer weights with a scaled normal distribution."""
    weights, biases = [], []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        scale = np.sqrt(2.0 / (in_size + out_size))
        weight_matrix = np.random.randn(in_size, out_size) * scale
        bias_vector = (
            np.random.randn(1, out_size) * scale if use_bias else np.zeros((1, out_size))
        )
        weights.append(weight_matrix)
        biases.append(bias_vector)
    return weights, biases


def activation(z, func):
    if func == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    if func == "tanh":
        return np.tanh(z)
    raise ValueError(f"Unsupported activation function: {func}")


def activation_derivative(a, func):
    if func == "sigmoid":
        return a * (1.0 - a)
    if func == "tanh":
        return 1.0 - a**2
    raise ValueError(f"Unsupported activation function: {func}")


def forward_pass(x, weights, biases, func):
    x = x.reshape(1, -1)
    nets = []
    outputs = [x]
    for weight_matrix, bias_vector in zip(weights, biases):
        net_input = outputs[-1] @ weight_matrix + bias_vector
        layer_output = activation(net_input, func)
        nets.append(net_input)
        outputs.append(layer_output)
    return nets, outputs


def build_layer_sizes(n_features, hidden_layers, n_classes):
    return [n_features, *hidden_layers, n_classes]
