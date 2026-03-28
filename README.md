# Bird Species Classifier — Backpropagation Neural Network

## Setup

```bash
pip install -r requirements.txt
```

Place the birds dataset at `data/penguins.csv`.  
Expected format: `feature1,feature2,feature3,feature4,feature5,class_label` (one row per sample, 50 rows per class, 3 classes).

---

## Who owns what

| File | Person | Status |
|------|--------|--------|
| `nn_core.py` | Person 1 | ⬜ TODO |
| `trainer.py` | Person 2 | ⬜ TODO |
| `evaluator.py` | Person 3 | ⬜ TODO |
| `main.py` | Person 3 | ⬜ TODO |

---

## Working independently (before integration)

Each file has a **self-test block** at the bottom and **mock stubs** for dependencies that aren't ready yet.

```bash
# Person 1 — test nn_core in isolation
python nn_core.py

# Person 2 — test trainer with mock forward pass (no Person 1 needed yet)
python trainer.py

# Person 3 — test evaluator with mock forward pass (no Person 1 needed yet)
python evaluator.py
```

When Person 1 finishes `nn_core.py`, Person 2 and 3 delete their mock stubs and uncomment the real imports at the top of their files.

---

## Shared interface contract

These are the function signatures everyone codes to. **Do not change signatures without telling the team.**

```python
# nn_core.py (Person 1)
forward_pass(x, weights, biases, func)         → (nets, outputs)
activation_derivative(a, func)                  → np.ndarray
init_weights(layer_sizes, use_bias)             → (weights, biases)
load_data(filepath, ...)                        → (X_train, y_train, X_test, y_test)
build_layer_sizes(n_features, hidden, n_classes)→ list[int]

# trainer.py (Person 2)
train(X_train, y_train, weights, biases, config)→ (weights, biases, accuracy_log)

# evaluator.py (Person 3)
confusion_matrix(X_test, y_test, weights, biases, func) → np.ndarray
overall_accuracy(matrix)                                → float
```

The `config` dict passed to `train()` always has these keys:
```python
{
    'eta'        : float,   # learning rate
    'epochs'     : int,
    'activation' : str,     # 'sigmoid' or 'tanh'
    'use_bias'   : bool,
    'hidden_layers': list,  # e.g. [4, 4]
}
```

---

## Integration checklist

Run this in order once all individual TODOs are done:

1. Person 2: delete mock `forward_pass` and `activation_derivative` in `trainer.py`, uncomment `from nn_core import ...`
2. Person 3: delete mock `forward_pass` in `evaluator.py`, uncomment `from nn_core import ...`
3. Person 3: uncomment all steps in `main.py` → `run()`, then uncomment `get_user_input()` call
4. Everyone: run `python main.py` end-to-end
5. Person 3: run experiments with both activations, record results in the report table

---

## Report table template (Task 2)

| Activation | Train Acc | Test Acc | LR | Epochs | #Layers | Hidden nodes |
|------------|-----------|----------|----|--------|---------|--------------|
| Sigmoid    |           |          |    |        |         |              |
| Tanh       |           |          |    |        |         |              |
