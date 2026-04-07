import numpy as np

from nn_core import forward_pass


def classify(x, weights, biases, func):
    """Predict class index and one-hot vector for a single sample."""
    _, outputs = forward_pass(x, weights, biases, func)
    logits = outputs[-1].flatten()
    predicted_class = int(np.argmax(logits))
    predicted_one_hot = np.zeros(len(logits))
    predicted_one_hot[predicted_class] = 1
    return predicted_one_hot, predicted_class


def confusion_matrix(X_test, y_test, weights, biases, func, n_classes=3):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for sample_idx in range(len(X_test)):
        true_class = int(np.argmax(y_test[sample_idx]))
        _, predicted_class = classify(X_test[sample_idx], weights, biases, func)
        matrix[true_class, predicted_class] += 1
    return matrix


def overall_accuracy(matrix):
    correct = np.trace(matrix)
    total = matrix.sum()
    return correct / total * 100.0


def print_results(matrix, train_accuracy, test_accuracy, config):
    print("Model Evaluation Summary")
    print("Confusion matrix (rows = actual, cols = predicted)")
    header = "        " + "  ".join(
        f"Class {class_id}" for class_id in range(matrix.shape[0])
    )
    print(header)
    for class_id, row in enumerate(matrix):
        print(
            f"  Class {class_id} " + "    ".join(str(value).rjust(3) for value in row)
        )
    print(f"\nTrain accuracy: {train_accuracy:.1f}%")
    print(f"Test accuracy : {test_accuracy:.1f}%")
    print(f"\nRun config: {config}")
