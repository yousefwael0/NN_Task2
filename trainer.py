"""
trainer.py — Person 2
=====================
Responsibilities:
  - Implement the backward pass (error signals per neuron)
  - Implement weight + bias updates
  - Run the full training loop (m epochs over n samples)
  - Track and return training accuracy per epoch

Imports from nn_core.py:
  forward_pass, activation_derivative

While Person 1 is still finishing nn_core.py, use the MOCK STUBS at the
bottom of this file to develop and test independently.
"""

import numpy as np

# --- real imports (switch to these once Person 1 is done) -----------------
# from nn_core import forward_pass, activation_derivative

# --- temporary mock stubs (delete these once nn_core.py is ready) ---------
def forward_pass(x, weights, biases, func):
    """MOCK — replace with: from nn_core import forward_pass"""
    x = x.reshape(1, -1)
    nets, outputs = [], [x]
    for w, b in zip(weights, biases):
        z = outputs[-1] @ w + b
        # mock sigmoid
        a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        nets.append(z)
        outputs.append(a)
    return nets, outputs

def activation_derivative(a, func):
    """MOCK — replace with: from nn_core import activation_derivative"""
    if func == 'sigmoid':
        return a * (1.0 - a)
    return 1.0 - a ** 2
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. BACKWARD PASS
# ---------------------------------------------------------------------------

def backward_pass(d: np.ndarray, outputs: list, weights: list,
                  func: str) -> list:
    """
    Compute the error (delta) for every neuron, propagating from output → input.

    Parameters
    ----------
    d       : desired output, shape (1, n_classes)  — one-hot encoded
    outputs : list of layer outputs from forward_pass()  [a0, a1, ..., a_L]
    weights : list of weight matrices
    func    : activation function name

    Returns
    -------
    deltas : list of delta arrays, one per layer (excluding the input layer)
             deltas[-1] is the output layer delta
             deltas[0]  is the first hidden layer delta
    """
    # TODO (Person 2): Implement backpropagation.
    #
    # Step 1 — Output layer delta:
    #   error  = d - outputs[-1]                         (shape: 1 × n_classes)
    #   delta  = error * activation_derivative(outputs[-1], func)
    #
    # Step 2 — Hidden layer deltas (loop backwards through layers):
    #   for each hidden layer l (from second-to-last down to first):
    #     error  = deltas[l+1] @ weights[l+1].T
    #     delta  = error * activation_derivative(outputs[l], func)
    #
    # Hint:
    #   n_layers = len(weights)            # number of weight matrices
    #   deltas   = [None] * n_layers
    #
    #   output_error     = d - outputs[-1]
    #   deltas[-1]       = output_error * activation_derivative(outputs[-1], func)
    #
    #   for l in range(n_layers - 2, -1, -1):
    #       error      = deltas[l + 1] @ weights[l + 1].T
    #       deltas[l]  = error * activation_derivative(outputs[l + 1], func)
    #
    #   return deltas

    raise NotImplementedError("TODO (Person 2): implement backward_pass()")


# ---------------------------------------------------------------------------
# 2. WEIGHT UPDATE
# ---------------------------------------------------------------------------

def update_weights(weights: list, biases: list, deltas: list,
                   outputs: list, eta: float, use_bias: bool) -> tuple[list, list]:
    """
    Apply the delta rule to update weights (and biases) for one sample.

    Rule:  w_new = w_old + eta * delta * input
           b_new = b_old + eta * delta          (if use_bias)

    Parameters
    ----------
    weights  : current weight matrices
    biases   : current bias vectors
    deltas   : error signals from backward_pass()
    outputs  : layer outputs from forward_pass()  (outputs[i] is input to layer i+1)
    eta      : learning rate
    use_bias : whether to update biases

    Returns
    -------
    Updated weights and biases (new lists — don't mutate in place)
    """
    # TODO (Person 2): Implement the weight update step.
    #
    #   for each layer l:
    #     weights[l] += eta * outputs[l].T @ deltas[l]
    #     if use_bias:
    #       biases[l]  += eta * deltas[l]
    #
    #   Return the modified weights and biases.
    #   (Since numpy arrays are mutable, updating in place is fine here,
    #    but return them for clarity.)
    #
    #   Hint:
    #   for l in range(len(weights)):
    #       weights[l] += eta * outputs[l].T @ deltas[l]
    #       if use_bias:
    #           biases[l] += eta * deltas[l]
    #   return weights, biases

    raise NotImplementedError("TODO (Person 2): implement update_weights()")


# ---------------------------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray,
          weights: list, biases: list, config: dict) -> tuple[list, list, list]:
    """
    Full training loop: m epochs over all training samples.

    Parameters
    ----------
    X_train  : shape (n_samples, n_features)
    y_train  : shape (n_samples, n_classes)  — one-hot
    weights  : initial weight matrices from nn_core.init_weights()
    biases   : initial bias vectors from nn_core.init_weights()
    config   : dict with keys:
                 'eta'        → float, learning rate
                 'epochs'     → int,   number of epochs (m)
                 'activation' → str,   'sigmoid' or 'tanh'
                 'use_bias'   → bool

    Returns
    -------
    weights        : trained weight matrices
    biases         : trained bias vectors
    accuracy_log   : list of float, training accuracy after each epoch
    """
    eta        = config['eta']
    epochs     = config['epochs']
    func       = config['activation']
    use_bias   = config['use_bias']
    n_samples  = X_train.shape[0]
    n_classes  = y_train.shape[1]

    accuracy_log = []

    for epoch in range(epochs):
        # TODO (Person 2): Implement the epoch loop.
        #
        # For each sample i = 0 → n_samples - 1:
        #   1. Fetch x = X_train[i], d = y_train[i]  (reshape to (1, -1))
        #   2. forward_pass(x, weights, biases, func)
        #   3. backward_pass(d, outputs, weights, func)
        #   4. update_weights(weights, biases, deltas, outputs, eta, use_bias)
        #
        # After each epoch, compute training accuracy:
        #   - Run forward_pass on every sample
        #   - Predict: argmax of the output layer
        #   - Compare to argmax of y_train[i]
        #   - accuracy = correct / n_samples
        #
        # Hint — inner loop skeleton:
        #
        # for i in range(n_samples):
        #     x = X_train[i].reshape(1, -1)
        #     d = y_train[i].reshape(1, -1)
        #     nets, outputs = forward_pass(x, weights, biases, func)
        #     deltas        = backward_pass(d, outputs, weights, func)
        #     weights, biases = update_weights(weights, biases, deltas, outputs, eta, use_bias)
        #
        # Hint — accuracy after epoch:
        #
        # correct = 0
        # for i in range(n_samples):
        #     x = X_train[i].reshape(1, -1)
        #     _, outs = forward_pass(x, weights, biases, func)
        #     pred    = np.argmax(outs[-1])
        #     true    = np.argmax(y_train[i])
        #     correct += int(pred == true)
        # acc = correct / n_samples * 100
        # accuracy_log.append(acc)
        #
        # if (epoch + 1) % max(1, epochs // 10) == 0:
        #     print(f"Epoch {epoch+1}/{epochs}  Train acc: {acc:.1f}%")

        raise NotImplementedError("TODO (Person 2): implement the epoch loop")

    return weights, biases, accuracy_log


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly: python trainer.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== trainer self-test (uses mock forward/activation) ===")

    # Once TODOs are done, un-comment to verify the training loop runs.
    #
    # np.random.seed(42)
    # from nn_core import init_weights, build_layer_sizes
    # layer_sizes = build_layer_sizes(5, [4], 3)
    # w, b = init_weights(layer_sizes, use_bias=True)
    #
    # # Dummy data: 90 samples, 5 features, 3 classes
    # X_dummy = np.random.randn(90, 5)
    # y_dummy = np.eye(3)[np.repeat([0,1,2], 30)]   # 30 per class, one-hot
    #
    # cfg = {'eta': 0.01, 'epochs': 50, 'activation': 'sigmoid', 'use_bias': True}
    # w_trained, b_trained, log = train(X_dummy, y_dummy, w, b, cfg)
    # print(f"Final train accuracy: {log[-1]:.1f}%")
    # print(f"Accuracy log (every 5 epochs): {log[::5]}")

    print("Uncomment the self-test block above once TODOs are implemented.")
