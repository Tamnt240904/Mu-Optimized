import inspect
import json
import math

import torch
import torch.nn as nn

import u_optimize
import lam
from lam import idg_pdf, standard_ig
from utilss import compute_Q, run_insertion_deletion


class LinearImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1).matmul(self.weight)


class QuadraticImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.tensor([1.0, 0.25, 2.0, 0.5], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.shape[0], -1)
        return (self.weight * flat ** 2).sum(dim=1)


class ZeroImageModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1).sum(dim=1) * 0.0


def _linear_case():
    model = LinearImageModel().eval()
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    baseline = torch.zeros_like(x)
    return model, x, baseline


def _quadratic_case():
    model = QuadraticImageModel().eval()
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    baseline = torch.zeros_like(x)
    return model, x, baseline


def _mu(result):
    return torch.tensor([s.mu_k for s in result.steps])


def test_mu_vectors_are_simplex_members():
    model, x, baseline = _linear_case()
    methods = u_optimize.run_all_methods(
        model, x, baseline, N=8, tau=0.01, mu_iter=20)

    for result in methods:
        mu = _mu(result)
        assert torch.all(mu >= -1e-7), result.name
        assert torch.isclose(mu.sum(), torch.tensor(1.0), atol=1e-6), result.name


def test_idg_pdf_weights_are_output_change_pdf_only():
    model, x, baseline = _quadratic_case()
    result = idg_pdf(model, x, baseline, N=8)
    mu = _mu(result)
    delta_f = torch.tensor([s.delta_f_k for s in result.steps])
    expected = delta_f.abs() / delta_f.abs().sum()

    assert not torch.allclose(expected, torch.full_like(expected, 1.0 / len(expected)))
    assert torch.allclose(mu, expected, atol=1e-6)


def test_idg_pdf_falls_back_to_uniform_when_output_change_is_zero():
    _, x, baseline = _linear_case()
    result = idg_pdf(ZeroImageModel(), x, baseline, N=8)
    mu = _mu(result)

    assert torch.allclose(mu, torch.full_like(mu, 1.0 / len(mu)), atol=1e-6)


def test_shared_path_convention_matches_definition():
    model, x, baseline = _quadratic_case()
    N = 8
    delta_x, _, _, d_list, df_list, f_vals, _ = lam._straight_line_pass(
        model, x, baseline, N)
    step = delta_x / N

    expected_f = []
    expected_d = []
    for k in range(N + 1):
        gamma = baseline + (k / N) * delta_x
        expected_f.append(float(model(gamma)))
        if k < N:
            grad = 2.0 * model.weight.view_as(x) * gamma
            expected_d.append(float((grad * step).sum()))

    expected_df = [
        expected_f[k + 1] - expected_f[k] for k in range(N)
    ]

    assert torch.allclose(torch.tensor(f_vals), torch.tensor(expected_f), atol=1e-6)
    assert torch.allclose(torch.tensor(df_list), torch.tensor(expected_df), atol=1e-6)
    assert torch.allclose(torch.tensor(d_list), torch.tensor(expected_d), atol=1e-6)


def test_step_d_matches_delta_f_on_linear_model():
    model, x, baseline = _linear_case()
    _, _, _, d_list, df_list, _, _ = lam._straight_line_pass(
        model, x, baseline, N=8)
    d = torch.tensor(d_list)
    delta_f = torch.tensor(df_list)

    assert d.shape == (8,)
    assert delta_f.shape == (8,)
    assert torch.isfinite(d).all()
    assert torch.isfinite(delta_f).all()
    assert torch.allclose(d, delta_f, atol=1e-6)


def test_methods_return_same_shape_as_input():
    model, x, baseline = _linear_case()
    methods = [
        standard_ig(model, x, baseline, N=8),
        idg_pdf(model, x, baseline, N=8),
        u_optimize.mu_optimized_ig(model, x, baseline, N=8, n_iter=20),
    ]

    assert [m.name for m in methods] == ["IG", "IDG-PDF", "μ-Optimized"]
    for result in methods:
        assert result.attributions.shape == x.shape


def test_q_is_finite_and_bounded():
    model, x, baseline = _linear_case()
    methods = u_optimize.run_all_methods(
        model, x, baseline, N=8, tau=0.01, mu_iter=20)

    for result in methods:
        assert math.isfinite(result.Q), result.name
        assert -1e-7 <= result.Q <= 1.0 + 1e-7, result.name

        d = torch.tensor([s.d_k for s in result.steps])
        delta_f = torch.tensor([s.delta_f_k for s in result.steps])
        mu = _mu(result)
        q = compute_Q(d, delta_f, mu)
        assert math.isfinite(q)
        assert -1e-7 <= q <= 1.0 + 1e-7


def test_mu_optimized_objective_has_no_weight_magnitude_reward():
    forbidden_terms = [
        "lambda",
        "signal",
        "harvest",
        "abs(",
        ".abs()",
        "abs_d",
    ]
    for source in [
        inspect.getsource(u_optimize.optimize_mu),
        inspect.getsource(lam.optimize_mu),
    ]:
        assert "lam" not in inspect.signature(lam.optimize_mu).parameters
        for term in forbidden_terms:
            assert term not in source


def test_linear_model_completeness_holds():
    model, x, baseline = _linear_case()
    target = float(model(x) - model(baseline))
    methods = u_optimize.run_all_methods(
        model, x, baseline, N=8, tau=0.01, mu_iter=20)

    for result in methods:
        assert torch.isclose(
            result.attributions.sum(),
            torch.tensor(target),
            atol=1e-5,
        ), result.name


def test_json_output_contains_diagnostics():
    model, x, baseline = _linear_case()
    methods = u_optimize.run_all_methods(
        model, x, baseline, N=8, tau=0.01, mu_iter=20, lr=0.05)
    required = {
        "mu_l2_sq",
        "mu_entropy",
        "mu_max",
        "mu_min",
        "mu_active_count",
        "mu_sum",
        "weighted_residual_l1",
        "weighted_residual_l2",
    }

    for result in methods:
        payload = result.to_dict()
        assert required.issubset(payload), result.name

    mu_payload = methods[-1].to_dict()
    assert mu_payload["Q_history"]
    assert mu_payload["objective_history"]
    assert "final_objective" in mu_payload
    assert "n_iters" in mu_payload
    assert "tau" in mu_payload
    assert "learning_rate" in mu_payload
    json.dumps({"methods": {m.name: m.to_dict() for m in methods}})


def test_insertion_deletion_export_payload_is_json_ready():
    model, x, baseline = _linear_case()
    methods = u_optimize.run_all_methods(
        model, x, baseline, N=8, tau=0.01, mu_iter=20, lr=0.05)
    payload = run_insertion_deletion(
        model, x, baseline, methods, n_steps=4)

    u_optimize._validate_insertion_deletion_export(payload, methods, 4)
    assert payload["perturb_steps"] == 4
    assert payload["replacement"] == "baseline"
    assert len(payload["perturb_schedule"]) == 5
    assert set(payload["methods"]) == {m.name for m in methods}

    for method in methods:
        method_payload = payload["methods"][method.name]
        assert len(method_payload["insertion_curve"]) == 5
        assert len(method_payload["deletion_curve"]) == 5
        assert math.isfinite(method_payload["insertion_auc"])
        assert math.isfinite(method_payload["deletion_auc"])
        assert method_payload["insdel"] == (
            method_payload["insertion_auc"] - method_payload["deletion_auc"]
        )
        assert method.insdel is not None

    json.dumps({"insertion_deletion": payload})
