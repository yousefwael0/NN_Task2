"""
main.py — Streamlit GUI
=======================
Run with:  streamlit run main.py

This file is NOT part of the graded discussion.
It just wires everything together into a UI.
"""

import streamlit as st
import pandas as pd

from nn_core import load_data, init_weights, build_layer_sizes
from trainer import train
from evaluator import confusion_matrix, overall_accuracy

# ---------------------------------------------------------------------------
DATA_PATH = "data/penguins.csv"
N_FEATURES = 5
N_CLASSES = 3
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bird Classifier — NN", layout="wide")
st.title("🐦 Bird Species Classifier")
st.caption("Multi-layer Neural Network with Backpropagation")

# ---------------------------------------------------------------------------
# SIDEBAR — User Inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Network Configuration")

    n_hidden = st.number_input(
        "Number of hidden layers", min_value=1, max_value=5, value=1, step=1
    )

    neurons_per_layer = []
    for i in range(int(n_hidden)):
        n = st.number_input(
            f"Neurons in hidden layer {i + 1}",
            min_value=1,
            max_value=64,
            value=4,
            step=1,
            key=f"neurons_{i}",
        )
        neurons_per_layer.append(int(n))

    eta = st.number_input(
        "Learning rate (η)",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f",
    )

    epochs = st.number_input(
        "Number of epochs", min_value=1, max_value=10000, value=100, step=50
    )

    use_bias = st.checkbox("Add bias", value=True)

    activation = st.radio(
        "Activation function",
        options=["sigmoid", "tanh"],
        format_func=lambda x: (
            "Sigmoid" if x == "sigmoid" else "Hyperbolic Tangent (tanh)"
        ),
    )

    run_btn = st.button("Train & Evaluate", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
if not run_btn:
    st.info("Configure the network in the sidebar, then click **Train & Evaluate**.")
    st.stop()

config = {
    "eta": float(eta),
    "epochs": int(epochs),
    "activation": activation,
    "use_bias": use_bias,
    "hidden_layers": neurons_per_layer,
}

# --- Load data ---
try:
    X_train, y_train, X_test, y_test = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# --- Build + init ---
layer_sizes = build_layer_sizes(N_FEATURES, neurons_per_layer, N_CLASSES)
weights, biases = init_weights(layer_sizes, use_bias)

st.subheader("Training")
st.caption(f"Architecture: {' → '.join(str(s) for s in layer_sizes)}")

# --- Train ---
with st.spinner(f"Training for {int(epochs)} epochs…"):
    weights, biases, acc_log = train(X_train, y_train, weights, biases, config)

train_acc = acc_log[-1] if acc_log else 0.0

# Accuracy curve
if acc_log:
    st.line_chart(
        pd.DataFrame({"Training Accuracy (%)": acc_log}), use_container_width=True
    )

# --- Evaluate ---
st.subheader("Evaluation")

matrix = confusion_matrix(X_test, y_test, weights, biases, activation)
test_acc = overall_accuracy(matrix)

col1, col2 = st.columns(2)
col1.metric("Train Accuracy", f"{train_acc:.1f}%")
col2.metric("Test Accuracy", f"{test_acc:.1f}%")

# Confusion matrix
labels = [f"Class {i}" for i in range(N_CLASSES)]
cm_df = pd.DataFrame(matrix, index=labels, columns=labels)
cm_df.index.name = "Actual \\ Predicted"

st.write("**Confusion Matrix**")
st.dataframe(
    cm_df.style.highlight_max(axis=None, color="#d4edda"), use_container_width=True
)

# --- Report table row (copy-paste into the report) ---
st.subheader("Report Table Entry")
report_row = pd.DataFrame(
    [
        {
            "Activation": activation.capitalize(),
            "Train Acc": f"{train_acc:.1f}%",
            "Test Acc": f"{test_acc:.1f}%",
            "LR": eta,
            "Epochs": int(epochs),
            "#Layers": int(n_hidden),
            "Hidden nodes": str(neurons_per_layer),
            "Bias": "Yes" if use_bias else "No",
        }
    ]
)
st.dataframe(report_row, use_container_width=True, hide_index=True)
