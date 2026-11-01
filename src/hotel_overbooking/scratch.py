import numpy as np
from sklearn.compose import ColumnTransformer


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_scratch(
    x: np.ndarray, y: np.ndarray, c: float = 1.0, steps: int = 20000, rate: float = 0.5
) -> tuple[np.ndarray, float]:
    """Gradient descent on mean log loss plus an L2 penalty, scaled to match sklearn's C."""
    n, k = x.shape
    weights = np.zeros(k)
    intercept = 0.0
    penalty = 1.0 / (c * n)
    for _ in range(steps):
        error = sigmoid(x @ weights + intercept) - y
        weights -= rate * (x.T @ error / n + penalty * weights)
        intercept -= rate * error.mean()
    return weights, float(intercept)


def predict_scratch(x: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return sigmoid(x @ weights + intercept)


def prepare_matrix(preprocessor: ColumnTransformer, df, fit: bool = False) -> np.ndarray:
    matrix = preprocessor.fit_transform(df) if fit else preprocessor.transform(df)
    return np.asarray(matrix, dtype=float)
