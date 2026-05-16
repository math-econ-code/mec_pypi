import contextlib
import io
import unittest

import numpy as np

try:
    import gurobipy as grb
    from mec.dw import TUMatching, TUMatchingEstimation
except ImportError as exc:
    grb = None
    GUROBI_IMPORT_ERROR = exc


def _require_gurobi():
    if grb is None:
        raise unittest.SkipTest(f"Gurobi is not available: {GUROBI_IMPORT_ERROR}")
    try:
        model = grb.Model()
        model.Params.OutputFlag = 0
        model.dispose()
    except grb.GurobiError as exc:
        raise unittest.SkipTest(f"Gurobi is not available: {exc}") from exc


def _minimal_random_market(
    seed=0,
    population_size=18,
    match_base=2.0,
    lambda_low=0.5,
    lambda_high=1.5,
    shock_scale=0.1,
    outside_value=-5.0,
    outside_scale=0.01,
):
    rng = np.random.default_rng(seed)
    X, Y = 3, 3
    K = 2
    I, J = population_size, population_size

    x_i = np.tile(np.arange(X), I // X)
    y_j = np.tile(np.arange(Y), J // Y)
    rng.shuffle(x_i)
    rng.shuffle(y_j)

    delta_i_x = np.eye(X)[x_i]
    delta_j_y = np.eye(Y)[y_j]

    features_x_k = rng.normal(size=(X, K))
    features_y_k = rng.normal(size=(Y, K))
    phi_x_y_k = (features_x_k[:, None, :] - features_y_k[None, :, :]) ** 2
    lambda_k = rng.uniform(lambda_low, lambda_high, size=K)
    Phi_x_y = match_base + (phi_x_y_k * lambda_k[None, None, :]).sum(axis=2)

    eps_i_y = rng.normal(0.0, shock_scale, size=(I, Y))
    eta_x_j = rng.normal(0.0, shock_scale, size=(X, J))
    eps_i_0 = np.full(I, outside_value) + rng.normal(0.0, outside_scale, size=I)
    eta_0_j = np.full(J, outside_value) + rng.normal(0.0, outside_scale, size=J)

    eps_hat_i_y = rng.normal(0.0, shock_scale, size=(I, Y))
    eta_hat_x_j = rng.normal(0.0, shock_scale, size=(X, J))
    eps_hat_i_0 = np.full(I, outside_value) + rng.normal(0.0, outside_scale, size=I)
    eta_hat_0_j = np.full(J, outside_value) + rng.normal(0.0, outside_scale, size=J)

    return {
        "Phi_x_y": Phi_x_y,
        "phi_x_y_k": phi_x_y_k,
        "lambda_k": lambda_k,
        "eps_i_y": eps_i_y,
        "eta_x_j": eta_x_j,
        "eps_i_0": eps_i_0,
        "eta_0_j": eta_0_j,
        "eps_hat_i_y": eps_hat_i_y,
        "eta_hat_x_j": eta_hat_x_j,
        "eps_hat_i_0": eps_hat_i_0,
        "eta_hat_0_j": eta_hat_0_j,
        "delta_i_x": delta_i_x,
        "delta_j_y": delta_j_y,
    }


def _run_silent(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _aggregate_counts(pi_i_y, pi_x_j, delta_i_x, delta_j_y):
    mu_x_y = delta_i_x.T @ pi_i_y
    mu_x0 = delta_i_x.T @ (1.0 - pi_i_y.sum(axis=1))
    mu_0y = (1.0 - pi_x_j.sum(axis=0)) @ delta_j_y

    mu_x_y = np.rint(mu_x_y).astype(int)
    mu_x0 = np.rint(mu_x0).astype(int)
    mu_0y = np.rint(mu_0y).astype(int)

    return mu_x_y, mu_x0, mu_0y


def _two_way_center(matrix):
    return (
        matrix
        - matrix.mean(axis=1, keepdims=True)
        - matrix.mean(axis=0, keepdims=True)
        + matrix.mean()
    )


class TestDantzigWolfeMinimal(unittest.TestCase):

    def test_forward_and_estimation_consistency(self):
        _require_gurobi()
        data = _minimal_random_market()

        full_problem = TUMatching(
            data["Phi_x_y"],
            data["eps_i_y"],
            data["eta_x_j"],
            data["eps_i_0"],
            data["eta_0_j"],
            data["delta_i_x"],
            data["delta_j_y"],
        )
        pi_i_y, pi_x_j, _, _, full_model = _run_silent(full_problem.gurobi_solve)

        rroa_problem = TUMatching(
            data["Phi_x_y"],
            data["eps_i_y"],
            data["eta_x_j"],
            data["eps_i_0"],
            data["eta_0_j"],
            data["delta_i_x"],
            data["delta_j_y"],
        )
        history, *_ = _run_silent(rroa_problem.rroa_solve, max_iter=50)

        self.assertEqual(full_model.Status, grb.GRB.OPTIMAL)
        self.assertTrue(history)
        np.testing.assert_allclose(history[-1], full_model.ObjVal, atol=1e-5)
        np.testing.assert_allclose(
            data["delta_i_x"].T @ pi_i_y,
            pi_x_j @ data["delta_j_y"],
            atol=1e-6,
        )

        mu_x_y, mu_x0, mu_0y = _aggregate_counts(
            pi_i_y,
            pi_x_j,
            data["delta_i_x"],
            data["delta_j_y"],
        )
        mu_x_y = mu_x_y.astype(float) + 1e-10
        mu_x0 = mu_x0.astype(float) + 1e-10
        mu_0y = mu_0y.astype(float) + 1e-10

        estimation_full = TUMatchingEstimation(
            mu_x_y,
            mu_x0,
            mu_0y,
            data["phi_x_y_k"],
            eps_i_0=data["eps_hat_i_0"],
            eta_0_j=data["eta_hat_0_j"],
            eps_i_y=data["eps_hat_i_y"],
            eta_x_j=data["eta_hat_x_j"],
        )
        _, _, _, estimation_full_model = _run_silent(estimation_full.unrestricted_solve)

        estimation_rroa = TUMatchingEstimation(
            mu_x_y,
            mu_x0,
            mu_0y,
            data["phi_x_y_k"],
            eps_i_0=data["eps_hat_i_0"],
            eta_0_j=data["eta_hat_0_j"],
            eps_i_y=data["eps_hat_i_y"],
            eta_x_j=data["eta_hat_x_j"],
        )
        estimation_history, *_ = _run_silent(estimation_rroa.rroa_solve, max_iter=50)

        self.assertEqual(estimation_full_model.Status, grb.GRB.OPTIMAL)
        self.assertTrue(estimation_history)
        np.testing.assert_allclose(
            estimation_history[-1],
            estimation_full_model.ObjVal,
            atol=1e-5,
        )

    def test_nonparametric_estimation_uses_pair_dummies(self):
        _require_gurobi()
        data = _minimal_random_market(
            seed=2,
            population_size=90,
            match_base=0.0,
            lambda_low=0.1,
            lambda_high=0.5,
            shock_scale=0.5,
            outside_value=0.0,
            outside_scale=0.5,
        )

        full_problem = TUMatching(
            data["Phi_x_y"],
            data["eps_i_y"],
            data["eta_x_j"],
            data["eps_i_0"],
            data["eta_0_j"],
            data["delta_i_x"],
            data["delta_j_y"],
        )
        pi_i_y, pi_x_j, _, _, _ = _run_silent(full_problem.gurobi_solve)
        mu_x_y, mu_x0, mu_0y = _aggregate_counts(
            pi_i_y,
            pi_x_j,
            data["delta_i_x"],
            data["delta_j_y"],
        )

        estimation_full = TUMatchingEstimation(
            mu_x_y.astype(float) + 1e-10,
            mu_x0.astype(float) + 1e-10,
            mu_0y.astype(float) + 1e-10,
            None,
            eps_i_0=data["eps_hat_i_0"],
            eta_0_j=data["eta_hat_0_j"],
            eps_i_y=data["eps_hat_i_y"],
            eta_x_j=data["eta_hat_x_j"],
        )
        _, _, _, estimation_full_model = _run_silent(estimation_full.unrestricted_solve)

        estimation_rroa = TUMatchingEstimation(
            mu_x_y.astype(float) + 1e-10,
            mu_x0.astype(float) + 1e-10,
            mu_0y.astype(float) + 1e-10,
            None,
            eps_i_0=data["eps_hat_i_0"],
            eta_0_j=data["eta_hat_0_j"],
            eps_i_y=data["eps_hat_i_y"],
            eta_x_j=data["eta_hat_x_j"],
        )
        estimation_history, *_ = _run_silent(estimation_rroa.rroa_solve, max_iter=100)

        self.assertTrue(estimation_rroa.nonparametric)
        self.assertEqual(estimation_rroa.K, estimation_rroa.X * estimation_rroa.Y)
        np.testing.assert_array_equal(
            estimation_rroa.phi_x_y_k @ np.arange(estimation_rroa.K),
            np.arange(estimation_rroa.K).reshape((estimation_rroa.X, estimation_rroa.Y)),
        )
        self.assertEqual(estimation_full_model.Status, grb.GRB.OPTIMAL)
        self.assertTrue(estimation_history)
        np.testing.assert_allclose(
            estimation_history[-1],
            estimation_full_model.ObjVal,
            atol=1e-5,
        )

        Phi_hat_x_y = estimation_rroa.get_Phi_x_y()
        centered_rmse = np.sqrt(np.mean(
            (_two_way_center(Phi_hat_x_y) - _two_way_center(data["Phi_x_y"])) ** 2
        ))
        self.assertLess(centered_rmse, 0.2)


if __name__ == "__main__":
    unittest.main()
