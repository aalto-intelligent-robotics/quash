import numpy as np
from enum import Enum


class MetricType(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    CUSTOM = 2


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    div = float(np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
    if div == 0:
        div = 1e-6
    d = np.dot(a, b) / div
    if d > 1:
        d = 1
    if d < -1:
        d = -1
    return d


def nonlinearity(x):
    return np.exp(-1 * x)


# def weighed_dot(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
#     w = nonlinearity(w)
#     return float(np.sum(np.multiply(np.multiply(a, b), w)))


# def cosine_distance_var(a: np.ndarray, b: np.ndarray, var: np.ndarray) -> float:
#     inv_var = 1 - var
#     div = float(np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
#     if div == 0:
#         div = 1e-6
#     d = weighed_dot(a, b, inv_var) / div
#     if d > 1:
#         d = 1.0
#     if d < -1:
#         d = -1.0
#     return d


def weighed_dot(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(np.multiply(np.multiply(a, b), w)))


def cosine_distance_var(a: np.ndarray, b: np.ndarray, var: np.ndarray) -> float:
    inv_var = np.abs(max(np.max(var), 1) - var)
    dim = float(len(inv_var))
    w = inv_var / np.sum(inv_var) * dim
    w = np.clip(w, 0, 1)
    a_mod = np.multiply(a, w)
    b_mod = np.multiply(b, w)
    div = float(np.linalg.norm(a_mod, ord=2) * np.linalg.norm(b_mod, ord=2))
    if div == 0:
        div = 1e-6
    d = weighed_dot(a, b, w) / div
    if d > 1:
        d = 1.0
    if d < -1:
        d = -1.0
    return d


def matrix_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_ = A / np.linalg.norm(A, axis=1)[:, None]
    B_ = B / np.linalg.norm(B, axis=1)[:, None]
    res = np.matmul(A_, B_.T)
    return res


def matrix_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    B = np.asarray(B)

    a_sq = np.sum(A**2, axis=1, keepdims=True)
    b_sq = np.sum(B**2, axis=1, keepdims=True).T
    ab = np.dot(A, B.T)

    dists_squared = a_sq + b_sq - 2 * ab
    dists_squared = np.maximum(dists_squared, 0)

    return np.sqrt(dists_squared)
