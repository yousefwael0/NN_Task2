import numpy as np

from nn_core import forward_pass


def classify(x, weights, biases, func):
    _, outputs = forward_pass(x, weights, biases, func)
    raw = outputs[-1].flatten()
    class_id = int(np.argmax(raw))
    y_hat = np.zeros(len(raw))
    y_hat[class_id] = 1
    return y_hat, class_id


def confusion_matrix(X_test, y_test, weights, biases, func, n_classes=3):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(X_test)):
        true_class = int(np.argmax(y_test[i]))
        _, pred_class = classify(X_test[i], weights, biases, func)
        matrix[true_class][pred_class] += 1
    return matrix


def overall_accuracy(matrix):
    correct = np.trace(matrix)
    total = matrix.sum()
    return correct / total * 100.0


def print_results(matrix, train_accuracy, test_accuracy, config):
    print("=== Confusion Matrix (rows=actual, cols=predicted) ===")
    header = "        " + "  ".join(f"Class{i}" for i in range(matrix.shape[0]))
    print(header)
    for i, row in enumerate(matrix):
        print(f"  Class{i}  " + "    ".join(str(v).rjust(3) for v in row))
    print(f"\n  Train Accuracy : {train_accuracy:.1f}%")
    print(f"  Test  Accuracy : {test_accuracy:.1f}%")
    print(f"\n  Config: {config}")