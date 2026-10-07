import numpy as np
import pytest
from src.games import (actions, features, sparse_synergy, edges, uniform_projection,
                       sampled_projection, anchored_projection, evaluate,
                       theorem_values, first_max, published_matrix, additive_matrix_projection)


@pytest.mark.parametrize("n", [4, 5, 8, 10])
@pytest.mark.parametrize("topology", ["additive", "chain", "ring", "star", "complete"])
def test_analytic_predictions_against_exhaustive_and_lstsq(n, topology):
    a = actions(n)
    x = features(a, topology)
    r = sparse_synergy(a)
    gram = x.T@x/len(x)
    np.testing.assert_allclose(gram, np.eye(x.shape[1]), atol=1e-12)
    coef = uniform_projection(x, r)
    independent, rank = sampled_projection(x, r)
    np.testing.assert_allclose(coef, independent, atol=1e-12)
    assert rank == x.shape[1]
    exact = evaluate(r, x@coef)
    theory = theorem_values(n, edge_count=len(edges(n, topology)))
    assert exact["mse"] == pytest.approx(theory["mse"], abs=1e-12)
    assert exact["regret"] == pytest.approx(theory["regret"], abs=1e-12)
    assert exact["chosen_index"] == 0
    np.testing.assert_allclose(coef[1:n+1], theory["unary_coefficient"], atol=1e-12)


@pytest.mark.parametrize("n", [4, 8, 12])
@pytest.mark.parametrize("weight", [1., 2., 16.])
def test_anchor_against_explicit_weighted_least_squares(n, weight):
    a = actions(n)
    x = features(a)
    r = sparse_synergy(a)
    weights = np.ones(len(a))
    weights[-1] = weight
    direct, _ = sampled_projection(x*np.sqrt(weights[:, None]), r*np.sqrt(weights))
    fast = anchored_projection(x, r, len(a)-1, weight)
    np.testing.assert_allclose(fast, direct, atol=1e-12)
    if weight >= 2:
        assert evaluate(r, x@fast)["success"] == 1


def test_no_synergy_training_cannot_learn_unobserved_spike():
    a = actions(8)
    x = features(a)
    r = sparse_synergy(a)
    # Every row except the special action has an exactly additive cost target.
    coef, rank = sampled_projection(x[:-1], r[:-1])
    assert rank == 9
    np.testing.assert_allclose(x[:-1]@coef, r[:-1], atol=1e-12)
    assert first_max(x@coef) == 0
    assert abs((x@coef)[-1]-r[-1]) == pytest.approx(1)


def test_evaluation_reports_known_error_regret_and_tie():
    result = evaluate(np.array([0., 2., 1.]), np.array([3., 1., 0.]))
    assert result["mse"] == pytest.approx(11/3)
    assert result["regret"] == 2
    assert result["max_error"] == 3
    assert result["success"] == 0
    assert first_max(np.array([1., 1., 0.])) == 0


def test_matrix_control_is_the_published_matrix_and_additive_projection():
    r = published_matrix()
    np.testing.assert_array_equal(r, [[8,-12,-12],[-12,0,0],[-12,0,0]])
    p = additive_matrix_projection(r)
    assert evaluate(r.ravel(), p.ravel())["regret"] == 8
    assert np.mean(p-r, axis=0) == pytest.approx(np.zeros(3))
    assert np.mean(p-r, axis=1) == pytest.approx(np.zeros(3))


def test_strength_changes_policy_but_not_projection_error():
    a = actions(8)
    x = features(a)
    outcomes = [evaluate(sparse_synergy(a,k), x@uniform_projection(x,sparse_synergy(a,k))) for k in [.5,1.,1.5]]
    assert outcomes[0]["success"] == 1
    assert outcomes[1]["chosen_index"] == 0  # Declared tie rule.
    assert outcomes[2]["success"] == 0
    assert outcomes[0]["mse"] == pytest.approx(outcomes[2]["mse"])
