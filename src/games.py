"""Small, fully enumerable games. No external MARL implementation is reused."""
from itertools import combinations
import numpy as np


def actions(n):
    if not 1 <= n <= 20:
        raise ValueError("Enumeration supports 1 to 20 binary agents")
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)


def sparse_synergy(a, kappa=1.5):
    n = a.shape[1]
    return np.all(a == 1, axis=1).astype(float) - kappa * 2.0**(1-n) * a.sum(axis=1)


def edges(n, topology):
    if topology == "additive":
        return []
    if topology == "chain":
        return [(i, i+1) for i in range(n-1)]
    if topology == "ring":
        return sorted(set(tuple(sorted((i, (i+1) % n))) for i in range(n)))
    if topology == "star":
        return [(0, i) for i in range(1, n)]
    if topology == "complete":
        return list(combinations(range(n), 2))
    raise ValueError(f"Unknown topology: {topology}")


def features(a, topology="additive"):
    x = 2*a-1
    columns = [np.ones(len(a))] + [x[:, i] for i in range(a.shape[1])]
    columns += [x[:, i]*x[:, j] for i, j in edges(a.shape[1], topology)]
    return np.column_stack(columns)


def uniform_projection(x, rewards):
    """Requires a complete orthogonal Walsh design, verified by tests."""
    return x.T @ rewards / len(rewards)


def sampled_projection(x_train, y_train):
    """Learner only receives training features/labels; no evaluator oracle."""
    coef, _, rank, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return coef, int(rank)


def anchored_projection(x, rewards, anchor, weight):
    """Privileged weighted population fit via a rank-one Gram update."""
    if weight < 1:
        raise ValueError("Anchor weight must be at least 1")
    coef = uniform_projection(x, rewards)
    v = x[anchor]
    delta = (weight-1)*(rewards[anchor]-v@coef)/(len(rewards)+(weight-1)*(v@v))
    return coef + delta*v


def first_max(values, atol=1e-12):
    return int(np.flatnonzero(values >= np.max(values)-atol)[0])


def evaluate(rewards, prediction):
    selected = first_max(prediction)
    residual = prediction-rewards
    mse = float(np.mean(residual**2))
    variance = float(np.var(rewards))
    reward_range = float(np.ptp(rewards))
    optimum = float(np.max(rewards))
    regret = optimum-float(rewards[selected])
    return {
        "mse": mse, "normalized_mse": mse/variance if variance else 0.0,
        "range_normalized_rmse": mse**0.5/reward_range if reward_range else 0.0,
        "max_error": float(np.max(np.abs(residual))), "regret": regret,
        "chosen_index": selected, "chosen_reward": float(rewards[selected]),
        "optimal_reward": optimum, "success": int(regret <= 1e-12),
        "reward_range": reward_range, "reward_variance": variance,
    }


def theorem_values(n, kappa=1.5, edge_count=0):
    p = 2.0**(-n)
    return {
        "mse": p-(n+1+edge_count)*p*p,
        "regret": max(0.0, 1-2*kappa*n*p) if kappa >= 1 else 0.0,
        "unary_coefficient": (1-kappa)*p,
        "constant_coefficient": p-kappa*n*p,
    }


def published_matrix():
    """QTRAN matrix, also Table 2 left of Weighted QMIX (2020)."""
    return np.array([[8., -12., -12.], [-12., 0., 0.], [-12., 0., 0.]])


def additive_matrix_projection(matrix):
    return matrix.mean(axis=1, keepdims=True) + matrix.mean(axis=0, keepdims=True) - matrix.mean()
