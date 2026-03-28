"""
nn_core.py — Person 1
=====================
Responsibilities:
  - Load and split the birds dataset
  - Initialize weights and biases
  - Define activation functions and their derivatives
  - Implement the forward pass

This module is the foundation everything else builds on.
P2 (trainer.py) imports: forward_pass, activation, activation_derivative
P3 (main.py)   imports: load_data, forward_pass
"""

import numpy as np
import csv
import os


# ---------------------------------------------------------------------------
# 1. DATASET LOADING
# ---------------------------------------------------------------------------

def load_data(filepath: str, num_classes: int = 3, samples_per_class: int = 50,
              train_per_class: int = 30):
    """
    Load the birds CSV, split into train/test per class.

    Expected CSV format: each row = [f1, f2, f3, f4, f5, class_label]
    Class labels should be integers 0, 1, 2  (or 1, 2, 3 — see TODO below).

    Returns
    -------
    X_train : np.ndarray, shape (num_classes * train_per_class, 5)
    y_train : np.ndarray, shape (num_classes * train_per_class, num_classes)  — one-hot
    X_test  : np.ndarray, shape (num_classes * test_per_class, 5)
    y_test  : np.ndarray, shape (num_classes * test_per_class, num_classes)   — one-hot
    """
    # TODO (Person 1): Load the actual birds CSV file.
    #   1. Open filepath with csv.reader or np.genfromtxt.
    #   2. Separate rows by class label.
    #   3. Take the FIRST 30 rows of each class for training (non-repeated).
    #   4. Take the REMAINING 20 rows of each class for testing.
    #   5. Call one_hot() on the labels before returning.
    #
    #   Hint — skeleton:
    #
    #   data_by_class = {i: [] for i in range(num_classes)}
    #   with open(filepath, newline='') as f:
    #       reader = csv.reader(f)
    #       next(reader)  # skip header if present
    #       for row in reader:
    #           features = list(map(float, row[:5]))
    #           label    = int(row[5])          # adjust index if needed
    #           data_by_class[label].append(features)
    #
    #   X_train_parts, y_train_parts = [], []
    #   X_test_parts,  y_test_parts  = [], []
    #   for cls, rows in data_by_class.items():
    #       rows = np.array(rows)
    #       X_train_parts.append(rows[:train_per_class])
    #       y_train_parts.append([cls] * train_per_class)
    #       X_test_parts.append(rows[train_per_class:samples_per_class])
    #       y_test_parts.append([cls] * (samples_per_class - train_per_class))
    #
    #   X_train = np.vstack(X_train_parts)
    #   X_test  = np.vstack(X_test_parts)
    #   y_train = one_hot(np.concatenate(y_train_parts), num_classes)
    #   y_test  = one_hot(np.concatenate(y_test_parts),  num_classes)
    #   return X_train, y_train, X_test, y_test

    raise NotImplementedError("TODO (Person 1): implement load_data()")


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer class labels to one-hot encoded matrix."""
    n = len(labels)
    encoded = np.zeros((n, num_classes))
    for i, label in enumerate(labels):
        encoded[i, int(label)] = 1.0
    return encoded


# ---------------------------------------------------------------------------
# 2. WEIGHT INITIALISATION
# ---------------------------------------------------------------------------

def init_weights(layer_sizes: list[int], use_bias: bool) -> tuple[list, list]:
    """
    Initialise weights and biases as small random numbers.

    Parameters
    ----------
    layer_sizes : list of ints
        Full architecture including input and output layers.
        e.g. [5, 4, 3] = input(5) → hidden(4) → output(3)
    use_bias : bool

    Returns
    -------
    weights : list of np.ndarray
        weights[i] has shape (layer_sizes[i], layer_sizes[i+1])
    biases  : list of np.ndarray  (zeros if use_bias is False)
        biases[i] has shape (1, layer_sizes[i+1])
    """
    # TODO (Person 1): Initialise weights with small random values.
    #   Use np.random.randn(...) * 0.01 (or similar small scale).
    #   If use_bias is False, fill biases with zeros — the training
    #   loop in trainer.py will still reference them, so keep the
    #   structure identical regardless.
    #
    #   Hint:
    #   weights, biases = [], []
    #   for i in range(len(layer_sizes) - 1):
    #       w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
    #       b = np.random.randn(1, layer_sizes[i+1]) * 0.01 if use_bias else np.zeros((1, layer_sizes[i+1]))
    #       weights.append(w)
    #       biases.append(b)
    #   return weights, biases

    raise NotImplementedError("TODO (Person 1): implement init_weights()")


# ---------------------------------------------------------------------------
# 3. ACTIVATION FUNCTIONS
# ---------------------------------------------------------------------------

def activation(z: np.ndarray, func: str) -> np.ndarray:
    """
    Apply the chosen activation function element-wise.

    Parameters
    ----------
    z    : net input array (any shape)
    func : 'sigmoid' or 'tanh'
    """
    # TODO (Person 1): Implement both activation functions.
    #
    #   Sigmoid : f(z) = 1 / (1 + exp(-z))
    #   Tanh    : f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    #             equivalently: np.tanh(z)
    #
    #   Use np.clip(z, -500, 500) inside sigmoid to avoid overflow.
    #
    #   if func == 'sigmoid':
    #       return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    #   elif func == 'tanh':
    #       return np.tanh(z)
    #   else:
    #       raise ValueError(f"Unknown activation: {func}")

    raise NotImplementedError("TODO (Person 1): implement activation()")


def activation_derivative(a: np.ndarray, func: str) -> np.ndarray:
    """
    Derivative of the activation function, given its OUTPUT a = f(z).
    (We pass a, not z, to avoid recomputing f(z).)

    Sigmoid derivative : f'(z) = a * (1 - a)
    Tanh derivative    : f'(z) = 1 - a²
    """
    # TODO (Person 1): Implement both derivatives.
    #
    #   if func == 'sigmoid':
    #       return a * (1.0 - a)
    #   elif func == 'tanh':
    #       return 1.0 - a ** 2
    #   else:
    #       raise ValueError(f"Unknown activation: {func}")

    raise NotImplementedError("TODO (Person 1): implement activation_derivative()")


# ---------------------------------------------------------------------------
# 4. FORWARD PASS
# ---------------------------------------------------------------------------

def forward_pass(x: np.ndarray, weights: list, biases: list,
                 func: str) -> tuple[list, list]:
    """
    Propagate a single sample (or batch) through the network.

    Parameters
    ----------
    x       : input array, shape (1, n_features) or (n_features,)
    weights : list of weight matrices from init_weights()
    biases  : list of bias vectors from init_weights()
    func    : activation function name ('sigmoid' or 'tanh')

    Returns
    -------
    nets    : list of net-input arrays [z1, z2, ..., z_L]  (one per layer, excl. input)
    outputs : list of output arrays    [a0, a1, ..., a_L]
              outputs[0] is the input layer (x itself)
    """
    # TODO (Person 1): Implement the forward pass.
    #
    #   For each layer l:
    #     net[l] = outputs[l-1] @ weights[l] + biases[l]
    #     outputs[l] = activation(net[l], func)
    #
    #   Make sure x is 2-D before starting: x = x.reshape(1, -1)
    #
    #   nets, outputs = [], [x.reshape(1, -1)]
    #   for w, b in zip(weights, biases):
    #       z = outputs[-1] @ w + b
    #       a = activation(z, func)
    #       nets.append(z)
    #       outputs.append(a)
    #   return nets, outputs

    raise NotImplementedError("TODO (Person 1): implement forward_pass()")


# ---------------------------------------------------------------------------
# 5. NETWORK ARCHITECTURE BUILDER  (helper used by main.py)
# ---------------------------------------------------------------------------

def build_layer_sizes(n_features: int, hidden_layers: list[int],
                      n_classes: int) -> list[int]:
    """
    Combine input size, hidden layer sizes, and output size into one list.

    Example: build_layer_sizes(5, [4, 4], 3) → [5, 4, 4, 3]
    """
    return [n_features] + hidden_layers + [n_classes]


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly: python nn_core.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== nn_core self-test ===")

    # Once TODOs are done, un-comment and run to verify shapes.
    #
    # layer_sizes = build_layer_sizes(5, [4, 4], 3)
    # w, b = init_weights(layer_sizes, use_bias=True)
    # print("Layer sizes:", layer_sizes)
    # print("Weight shapes:", [wi.shape for wi in w])
    #
    # x_dummy = np.random.randn(1, 5)
    # nets, outs = forward_pass(x_dummy, w, b, func='sigmoid')
    # print("Output shape:", outs[-1].shape)   # should be (1, 3)
    # print("Output sum  :", outs[-1].sum())    # not softmax, so won't sum to 1
    #
    # print("Sigmoid(0)  :", activation(np.array([[0.0]]), 'sigmoid'))  # → 0.5
    # print("Tanh(0)     :", activation(np.array([[0.0]]), 'tanh'))     # → 0.0
    print("Uncomment the self-test block above once TODOs are implemented.")
