import streamlit as st
import pandas as pd
import numpy as np
from nn_core import load_data, init_weights, build_layer_sizes
from trainer import train
from evaluator import confusion_matrix, overall_accuracy

DATA_PATH = "data/penguins_preprocessed.csv"
N_FEATURES = 5
N_CLASSES = 3
CLASS_LABELS = ["Adelie", "Chinstrap", "Gentoo"]
np.random.seed(42)


def build_report_row(
    activation,
    train_acc,
    test_acc,
    learning_rate,
    epochs,
    n_hidden,
    hidden_nodes,
    use_bias,
):
    return pd.DataFrame(
        [
            {
                "Activation Function": activation.capitalize(),
                "Training Accuracy": f"{train_acc:.1f}%",
                "Testing Accuracy": f"{test_acc:.1f}%",
                "Learning Rate": learning_rate,
                "Epoch Count": int(epochs),
                "Hidden Layer Count": int(n_hidden),
                "Neurons per Hidden Layer": str(hidden_nodes),
                "Bias Enabled": "Yes" if use_bias else "No",
            }
        ]
    )


st.set_page_config(page_title="Penguin Species Classifier", layout="wide")
st.title("🐧 Penguin Species Classifier")
st.caption("Train and evaluate a small neural network built from scratch.")

with st.sidebar:
    st.header("Model Settings")

    n_hidden = st.number_input(
        "Hidden layers", min_value=1, max_value=5, value=1, step=1
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

    learning_rate = st.number_input(
        "Learning rate",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f",
    )

    epochs = st.number_input(
        "Training epochs", min_value=1, max_value=10000, value=100, step=50
    )

    use_bias = st.checkbox("Use bias terms", value=True)

    activation = st.radio(
        "Activation",
        options=["sigmoid", "tanh"],
        format_func=lambda x: ("Sigmoid" if x == "sigmoid" else "Tanh"),
    )

    run_btn = st.button("Train Model", type="primary", use_container_width=True)

if not run_btn:
    st.info("Set your model options in the sidebar, then click **Train Model**.")
    st.stop()

config = {
    "learning_rate": float(learning_rate),
    "epochs": int(epochs),
    "activation": activation,
    "use_bias": use_bias,
    "hidden_layers": neurons_per_layer,
}

try:
    X_train, y_train, X_test, y_test = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load the dataset. Details: {e}")
    st.stop()

layer_sizes = build_layer_sizes(N_FEATURES, neurons_per_layer, N_CLASSES)
weights, biases = init_weights(layer_sizes, use_bias)

st.subheader("Training Progress")
st.caption(f"Network shape: {' → '.join(str(size) for size in layer_sizes)}")

with st.spinner(f"Training for {int(epochs)} epochs. This may take a moment..."):
    weights, biases, acc_log = train(X_train, y_train, weights, biases, config)

train_acc = acc_log[-1] if acc_log else 0.0

if acc_log:
    st.line_chart(
        pd.DataFrame({"Training accuracy (%)": acc_log}), use_container_width=True
    )

st.subheader("Test Results")

matrix = confusion_matrix(X_test, y_test, weights, biases, activation)
test_acc = overall_accuracy(matrix)

col1, col2 = st.columns(2)
col1.metric("Final Train Accuracy", f"{train_acc:.1f}%")
col2.metric("Test Accuracy", f"{test_acc:.1f}%")

cm_df = pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS)
cm_df.index.name = "Actual \\ Predicted"

st.write("**Confusion Matrix**")
st.dataframe(
    cm_df.style.highlight_max(axis=None, color="#d4edda"), use_container_width=True
)

st.subheader("Run Summary")
st.caption("Use this row directly in your assignment report table.")
report_row = build_report_row(
    activation,
    train_acc,
    test_acc,
    learning_rate,
    epochs,
    n_hidden,
    neurons_per_layer,
    use_bias,
)
st.dataframe(report_row, use_container_width=True, hide_index=True)
