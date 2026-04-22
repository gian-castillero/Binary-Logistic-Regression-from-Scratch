# Binary Logistic Regression from Scratch

A ground-up implementation of binary logistic regression in Python using NumPy, matching the scikit-learn API and validated against the scikit-learn implementation on synthetic data.

## Overview

This project implements a fully functional `BinaryLogisticRegression` class that trains via gradient descent on the log-likelihood objective. The implementation follows the scikit-learn API pattern (`fit`, `predict`, `predict_proba`) and is benchmarked against scikit-learn's `LogisticRegression`. An ablation study explores how the learning rate and number of iterations interact to affect training performance.

## Implementation

The `BinaryLogisticRegression` class (in `logistic.ipynb`) supports:

- **`__init__(lr, iters, random_state)`** — Initializes weights from a standard normal distribution.
- **`predict_proba(X)`** — Computes sigmoid activation of the dot product of weights and input features.
- **`predict(X)`** — Thresholds probabilities at 0.5 to produce binary class labels.
- **`fit(X, y, verbose=False)`** — Runs gradient descent for `iters` iterations using the gradient of the log-likelihood loss.

### Gradient Descent Update Rule

For each iteration, the gradient with respect to weight $w_j$ is:

$$\nabla_j = \frac{1}{n} \sum_{i=1}^{n} (a_i - y_i) x_{ij}$$

where $a_i = \sigma(\vec{w} \cdot \vec{x}_i)$ is the predicted probability. The weight update is:

$$\vec{w}' = \vec{w} - \eta \nabla$$

All operations are vectorized using NumPy for efficiency — no loops over individual data points.

## Validation

Trained on synthetic data (1000 samples, 20 features, `random_state=2024`) split 70/30.

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| scikit-learn LogisticRegression | 87.1% | 88.0% |
| DIY BinaryLogisticRegression | 86.7% | 87.7% |

The DIY implementation matches the scikit-learn baseline closely.

## Ablation Study: Learning Rate × Iterations

20 models were trained across all combinations of `lr ∈ {10, 1, 0.1, 0.01, 0.001}` and `iters ∈ {1, 5, 20, 100}`, evaluated on the training set:

| Learning Rate | 1 iter | 5 iters | 20 iters | 100 iters |
|--------------|--------|---------|----------|-----------|
| **10** | 0.843 | 0.867 | 0.869 | 0.867 |
| 1 | 0.464 | 0.681 | 0.853 | 0.869 |
| 0.1 | 0.561 | 0.530 | 0.423 | 0.771 |
| 0.01 | 0.569 | 0.504 | 0.457 | 0.484 |
| 0.001 | 0.417 | 0.420 | 0.504 | 0.514 |

**Best setting: `lr=10`** — achieves the highest accuracy across all iteration counts, and near-optimal accuracy in just 1 iteration, making it the most computationally efficient choice. Lower learning rates require many more iterations to converge, and very small rates (0.001, 0.01) fail to converge even at 100 iterations.

## Key Findings

- A pure NumPy logistic regression implementation can match scikit-learn's accuracy on synthetic data.
- Learning rate has a dramatic effect on convergence speed: too small a rate prevents convergence even with many iterations, while an appropriately large rate converges in very few steps.
- Vectorized NumPy operations (avoiding Python loops over data) are essential for practical performance.

## Tech Stack

- Python 3
- NumPy
- scikit-learn (for baseline comparison and accuracy scoring)

## How to Run

```bash
pip install scikit-learn numpy jupyter
jupyter notebook logistic.ipynb
```

No external data needed — synthetic data is generated in the notebook.
