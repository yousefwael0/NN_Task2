"""
evaluator.py — Person 3
=======================
Responsibilities:
  - Run the testing phase (forward pass → argmax → class ID)
  - Build the 3×3 confusion matrix
  - Compute and display overall accuracy

Imports from nn_core.py:
  forward_pass

While Person 1 is finishing nn_core.py, use the MOCK STUB below.
"""

import numpy as np

# --- real import (switch once Person 1 is done) ---------------------------
# from nn_core import forward_pass

# --- temporary mock stub (delete once nn_core.py is ready) ----------------
def forward_pass(x, weights, biases, func):
    """MOCK — replace with: from nn_core import forward_pass"""
    x = x.reshape(1, -1)
    nets, outputs = [], [x]
    for w, b in zip(weights, biases):
        z = outputs[-1] @ w + b
        a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        nets.append(z)
        outputs.append(a)
    return nets, outputs
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. CLASSIFY A SINGLE SAMPLE
# ---------------------------------------------------------------------------

def classify(x: np.ndarray, weights: list, biases: list,
             func: str) -> tuple[np.ndarray, int]:
    """
    Classify one sample: forward pass → argmax → one-hot output + class ID.

    Parameters
    ----------
    x       : feature vector, shape (n_features,) or (1, n_features)
    weights : trained weight matrices
    biases  : trained bias vectors
    func    : activation function name ('sigmoid' or 'tanh')

    Returns
    -------
    y_hat   : np.ndarray, shape (n_classes,) — one-hot predicted output
              e.g. [0, 1, 0] if class 2 was predicted
    class_id: int — index of the predicted class (0, 1, or 2)
    """
    # TODO (Person 3): Implement single-sample classification.
    #
    # 1. Call forward_pass(x, weights, biases, func)
    # 2. Take the final layer's output: raw_output = outputs[-1].flatten()
    # 3. class_id = np.argmax(raw_output)
    # 4. Build y_hat: all zeros, set y_hat[class_id] = 1
    # 5. Return y_hat, class_id
    #
    # Hint:
    #   _, outputs = forward_pass(x, weights, biases, func)
    #   raw        = outputs[-1].flatten()
    #   class_id   = int(np.argmax(raw))
    #   y_hat      = np.zeros(len(raw))
    #   y_hat[class_id] = 1
    #   return y_hat, class_id

    raise NotImplementedError("TODO (Person 3): implement classify()")


# ---------------------------------------------------------------------------
# 2. CONFUSION MATRIX
# ---------------------------------------------------------------------------

def confusion_matrix(X_test: np.ndarray, y_test: np.ndarray,
                     weights: list, biases: list,
                     func: str, n_classes: int = 3) -> np.ndarray:
    """
    Build the n_classes × n_classes confusion matrix over the test set.

    confusion[true_class][predicted_class] += 1

    Parameters
    ----------
    X_test    : shape (n_test_samples, n_features)
    y_test    : shape (n_test_samples, n_classes)  — one-hot
    weights   : trained weight matrices
    biases    : trained bias vectors
    func      : activation function name
    n_classes : number of output classes

    Returns
    -------
    matrix : np.ndarray, shape (n_classes, n_classes), dtype int
    """
    # TODO (Person 3): Build the confusion matrix.
    #
    # matrix = np.zeros((n_classes, n_classes), dtype=int)
    # for i in range(len(X_test)):
    #     true_class = int(np.argmax(y_test[i]))
    #     _, pred_class = classify(X_test[i], weights, biases, func)
    #     matrix[true_class][pred_class] += 1
    # return matrix

    raise NotImplementedError("TODO (Person 3): implement confusion_matrix()")


# ---------------------------------------------------------------------------
# 3. OVERALL ACCURACY
# ---------------------------------------------------------------------------

def overall_accuracy(matrix: np.ndarray) -> float:
    """
    Compute overall accuracy from the confusion matrix.

    accuracy = sum of diagonal / sum of all cells  (as a percentage)
    """
    # TODO (Person 3): Compute accuracy.
    #
    # correct = np.trace(matrix)          # sum of diagonal (true positives)
    # total   = matrix.sum()
    # return correct / total * 100.0

    raise NotImplementedError("TODO (Person 3): implement overall_accuracy()")


# ---------------------------------------------------------------------------
# 4. PRETTY PRINT
# ---------------------------------------------------------------------------

def print_results(matrix: np.ndarray, train_accuracy: float,
                  test_accuracy: float, config: dict) -> None:
    """
    Display the confusion matrix and accuracy summary in a readable format.

    Parameters
    ----------
    matrix         : confusion matrix from confusion_matrix()
    train_accuracy : final training accuracy (%) from trainer.py
    test_accuracy  : test accuracy (%) computed here
    config         : the config dict used for this run (for the report table)
    """
    # TODO (Person 3): Print a nicely formatted summary.
    #
    # Suggested layout:
    #
    # ┌─────────────────────────────────────────────┐
    # │  Confusion Matrix (rows=actual, cols=pred)  │
    # │        Class0  Class1  Class2               │
    # │  Class0   xx      xx      xx                │
    # │  Class1   xx      xx      xx                │
    # │  Class2   xx      xx      xx                │
    # │                                             │
    # │  Train Accuracy : xx.x%                     │
    # │  Test  Accuracy : xx.x%                     │
    # │                                             │
    # │  Config: LR=xx  Epochs=xx  Activation=xx   │
    # │          Layers=xx  Neurons=xx  Bias=xx     │
    # └─────────────────────────────────────────────┘
    #
    # Hint — minimal version:
    #
    # print("\n=== Confusion Matrix (rows=actual, cols=predicted) ===")
    # header = "        " + "  ".join(f"Class{i}" for i in range(matrix.shape[0]))
    # print(header)
    # for i, row in enumerate(matrix):
    #     print(f"  Class{i}  " + "    ".join(str(v).rjust(3) for v in row))
    # print(f"\n  Train Accuracy : {train_accuracy:.1f}%")
    # print(f"  Test  Accuracy : {test_accuracy:.1f}%")
    # print(f"\n  Config: {config}")

    raise NotImplementedError("TODO (Person 3): implement print_results()")


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly: python evaluator.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== evaluator self-test (uses mock forward pass) ===")

    # Once TODOs are done, un-comment to verify shapes and output.
    #
    # np.random.seed(0)
    # # Fake weights for a 5→4→3 network
    # w = [np.random.randn(5, 4) * 0.1, np.random.randn(4, 3) * 0.1]
    # b = [np.zeros((1, 4)),            np.zeros((1, 3))]
    #
    # # 60 test samples (20 per class), 5 features
    # X_test  = np.random.randn(60, 5)
    # y_test  = np.eye(3)[np.repeat([0, 1, 2], 20)]
    #
    # mat = confusion_matrix(X_test, y_test, w, b, func='sigmoid')
    # acc = overall_accuracy(mat)
    # cfg = {'eta': 0.01, 'epochs': 100, 'activation': 'sigmoid',
    #        'use_bias': False, 'hidden_layers': [4]}
    # print_results(mat, train_accuracy=65.0, test_accuracy=acc, config=cfg)

    print("Uncomment the self-test block above once TODOs are implemented.")
