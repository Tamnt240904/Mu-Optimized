"""
u_optimize.py — μ-Optimized Integrated Gradients
================================================

Implements μ-optimization for discretized Integrated Gradients:

    min_{μ∈simplex}  -Q(μ) +  (τ/2) ‖μ‖²₂

Compares three methods:
    1. Standard IG      — uniform μ, straight line
    2. IDG-PDF          — μ_k ∝ |Δf_k|, straight line
    3. μ-Optimized      — optimized μ via projected gradient descent

Usage:
    from u_optimize import mu_optimized_ig, run_all_methods, run_experiment
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn

# Import shared infrastructure from the existing codebase
from utilss import (
    AttributionResult, StepInfo,
    compute_Var_nu, compute_CV2, compute_Q, compute_all_metrics,
    compute_residual_diagnostics, compute_weight_diagnostics,
    completeness_rescale,
)

# Import path/gradient utilities from the existing experiment infrastructure
from lam import (
    _forward_scalar, _forward_batch, _forward_and_gradient,
    _forward_and_gradient_batch, _gradient, _gradient_batch,
    _dot, _rescale, _build_steps, _straight_line_pass, optimize_mu,
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  μ OBJECTIVE
# ═════════════════════════════════════════════════════════════════════════════

def compute_mu_objective(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    mu: torch.Tensor,
    tau: float = 0.01,
) -> tuple[float, float, float]:
    """
    Evaluate the μ-optimized IG objective:

        L(μ) = -Q(μ) + (τ/2) ‖μ‖²₂

    Args:
        d:       (N,) tensor of d_k = ∇f(γ_k) · ((x - baseline)/N)
        delta_f: (N,) tensor of Δf_k = f(γ_{k+1}) − f(γ_k)
        mu:      (N,) probability measure over steps
        tau:     L2 admissibility multiplier τ

    Returns:
        (total_objective, q_term, l2_term)
    """
    q = compute_Q(d, delta_f, mu)
    l2 = float((mu ** 2).sum())
    total = -q + (tau / 2.0) * l2
    return total, q, l2


def output_change_weights(
    d: torch.Tensor,
    delta_f: torch.Tensor,
) -> torch.Tensor:
    """Closed-form IDG-PDF heuristic μ_k ∝ |Δf_k|."""
    del d
    weights = delta_f.abs()
    w_sum = weights.sum()
    if w_sum < 1e-12:
        return torch.full_like(weights, 1.0 / len(weights))
    return weights / w_sum


# ═════════════════════════════════════════════════════════════════════════════
# §2  μ-OPTIMIZED IG
# ═════════════════════════════════════════════════════════════════════════════

def _pack_result(name, attr, d_list, df_list, f_vals, gnorms, mu, N,
                 t0, Q_history=None, objective_history=None,
                 optimizer_info=None) -> AttributionResult:
    """Build AttributionResult with all metrics."""
    device = attr.device
    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)
    var_nu, cv2, Q = compute_all_metrics(d_arr, df_arr, mu)
    steps = _build_steps(d_list, df_list, f_vals, gnorms, mu, N)
    diagnostics = {}
    diagnostics.update(compute_weight_diagnostics(mu))
    diagnostics.update(compute_residual_diagnostics(d_arr, df_arr, mu))
    return AttributionResult(
        name=name, attributions=attr, Q=Q, CV2=cv2, Var_nu=var_nu,
        steps=steps, Q_history=Q_history or [],
        objective_history=objective_history or [],
        diagnostics=diagnostics,
        optimizer_info=optimizer_info or {},
        elapsed_s=time.time() - t0,
    )


def mu_optimized_ig(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    tau: float = 0.01,
    n_iter: int = 300,
    lr: float = 0.05,
    log_every: int = 20,
) -> AttributionResult:
    """
    Straight line + μ optimized under -Q(μ) + (τ/2)||μ||²₂.

    Cost: standard IG + O(N) arithmetic.  Zero extra model evaluations.
    """
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    d_arr = torch.tensor(d_list, device=x.device)
    df_arr = torch.tensor(df_list, device=x.device)

    mu, history = optimize_mu(
        d_arr, df_arr, tau=tau, n_iter=n_iter, lr=lr,
        log_every=log_every, return_history=True)

    # Weighted gradient sum
    grad_stack = torch.cat(grads, dim=0)                 # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)   # (1, C, H, W)
    attr = completeness_rescale(delta_x * wg, target)

    optimizer_info = {
        "final_objective": history["final_objective"],
        "n_iters": history["n_iters"],
        "tau": history["tau"],
        "learning_rate": history["learning_rate"],
    }
    return _pack_result(
        "μ-Optimized", attr, d_list, df_list, f_vals, gnorms, mu, N, t0,
        Q_history=history["Q_history"],
        objective_history=history["objective_history"],
        optimizer_info=optimizer_info)


# ═════════════════════════════════════════════════════════════════════════════
# §3  RUN ALL METHODS (IG, IDG-PDF, μ-Optimized)
# ═════════════════════════════════════════════════════════════════════════════

def run_all_methods(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    tau: float = 0.01,
    mu_iter: int = 300,
    lr: float = 0.05,
) -> list[AttributionResult]:
    """
    Run three IG variants: IG, IDG-PDF, and μ-Optimized.

    Returns list: [IG, IDG-PDF, μ-Optimized]
    """
    from lam import standard_ig, idg_pdf

    results = []

    # 1. Standard IG  (uniform μ, straight line)
    results.append(standard_ig(model, x, baseline, N))

    # 2. IDG-PDF  (μ_k ∝ |Δf_k|, straight line)
    results.append(idg_pdf(model, x, baseline, N))

    # 3. μ-Optimized  (straight line, optimized μ)
    results.append(mu_optimized_ig(
        model, x, baseline, N, tau=tau, n_iter=mu_iter, lr=lr))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# §4  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(N=50, device=None, min_conf=0.70, tau=0.01,
                   mu_iter=300, lr=0.05, skip=0):
    """Run experiment: load model/image, run 3 methods and print table."""
    from lam import load_image_and_model
    from utilss import get_device

    if device is None:
        device = get_device()

    print("Loading ResNet-50 and image...")
    model, x, baseline, info = load_image_and_model(device, min_conf, skip=skip)

    f_x = _forward_scalar(model, x)
    f_bl = _forward_scalar(model, baseline)
    delta_f = f_x - f_bl

    print(f"\nModel : {info['model']}")
    print(f"Source: {info['source']}")
    print(f"Class : {info['target_class']} (conf={info['confidence']:.4f})")
    print(f"f(x) = {f_x:.4f},  f(bl) = {f_bl:.4f},  Δf = {delta_f:.4f}")
    print(f"N = {N},  τ = {tau},  iters = {mu_iter},  lr = {lr}\n")

    methods = run_all_methods(
        model, x, baseline, N=N, tau=tau, mu_iter=mu_iter, lr=lr)

    # ── Print table ──
    hdr = (f"{'Method':<16} {'Var_ν':>10} {'CV²':>8} {'𝒬':>8} "
           f"{'Obj':>10} {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    for m in methods:
        d_arr = torch.tensor([s.d_k for s in m.steps], device=device)
        df_arr = torch.tensor([s.delta_f_k for s in m.steps], device=device)
        mu_arr = torch.tensor([s.mu_k for s in m.steps], device=device)
        obj, _, _ = compute_mu_objective(d_arr, df_arr, mu_arr, tau=tau)
        print(f"{m.name:<16} {m.Var_nu:>10.6f} {m.CV2:>8.4f} "
              f"{m.Q:>8.4f} {obj:>10.4f} {m.elapsed_s:>7.1f}s")

    results = {
        "config": {"N": N, "tau": tau, "iters": mu_iter, "lr": lr},
        "image_info": info,
        "model_info": {"f_x": f_x, "f_baseline": f_bl,
                       "delta_f": delta_f, "N": N, "device": str(device)},
        "methods": {m.name: m.to_dict() for m in methods},
    }
    return results, methods, model, x, baseline, info


def _run_loaded_experiment(model, x, baseline, info, device, N, tau,
                           mu_iter, lr):
    f_x = _forward_scalar(model, x)
    f_bl = _forward_scalar(model, baseline)
    delta_f = f_x - f_bl
    methods = run_all_methods(
        model, x, baseline, N=N, tau=tau, mu_iter=mu_iter, lr=lr)
    return {
        "config": {"N": N, "tau": tau, "iters": mu_iter, "lr": lr},
        "image_info": info,
        "model_info": {"f_x": f_x, "f_baseline": f_bl,
                       "delta_f": delta_f, "N": N, "device": str(device)},
        "methods": {m.name: m.to_dict() for m in methods},
    }


def _validate_insertion_deletion_export(
    payload: dict,
    methods: list[AttributionResult],
    insdel_steps: int,
) -> None:
    if insdel_steps <= 0:
        raise ValueError(f"insdel_steps must be positive, got {insdel_steps}")

    expected_len = insdel_steps + 1
    method_payloads = payload.get("methods", {})
    if payload.get("perturb_steps") != insdel_steps:
        raise ValueError(
            "insertion/deletion export has inconsistent perturb_steps: "
            f"{payload.get('perturb_steps')} != {insdel_steps}"
        )

    schedule = payload.get("perturb_schedule")
    if not isinstance(schedule, list) or len(schedule) != expected_len:
        raise ValueError(
            "insertion/deletion export has invalid perturb_schedule length: "
            f"{None if schedule is None else len(schedule)} != {expected_len}"
        )

    expected_schedule = [float(i) / insdel_steps for i in range(expected_len)]
    if any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(schedule, expected_schedule)
    ):
        raise ValueError("insertion/deletion export perturb_schedule is invalid")

    for method in methods:
        if method.name not in method_payloads:
            raise ValueError(
                f"insertion/deletion export missing method {method.name!r}")
        method_payload = method_payloads[method.name]
        if not isinstance(method_payload, dict):
            raise ValueError(f"{method.name}: insertion/deletion payload is invalid")
        insertion_curve = method_payload.get("insertion_curve")
        deletion_curve = method_payload.get("deletion_curve")
        if not isinstance(insertion_curve, list):
            raise ValueError(f"{method.name}: insertion_curve is not a list")
        if not isinstance(deletion_curve, list):
            raise ValueError(f"{method.name}: deletion_curve is not a list")
        if len(insertion_curve) != expected_len:
            raise ValueError(
                f"{method.name}: insertion_curve length "
                f"{len(insertion_curve)} != {expected_len}"
            )
        if len(deletion_curve) != expected_len:
            raise ValueError(
                f"{method.name}: deletion_curve length "
                f"{len(deletion_curve)} != {expected_len}"
            )
        insertion_auc = method_payload.get("insertion_auc")
        deletion_auc = method_payload.get("deletion_auc")
        if not isinstance(insertion_auc, (int, float)) or not math.isfinite(insertion_auc):
            raise ValueError(f"{method.name}: insertion_auc is not finite")
        if not isinstance(deletion_auc, (int, float)) or not math.isfinite(deletion_auc):
            raise ValueError(f"{method.name}: deletion_auc is not finite")


# ═════════════════════════════════════════════════════════════════════════════
# §5  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json
    from utilss import set_seed
    parser = argparse.ArgumentParser(
        description="μ-Optimized IG — compare IG, IDG-PDF, and μ-Optimized")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of interpolation steps N")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="L2 admissibility multiplier τ")
    parser.add_argument("--iters", type=int, default=300,
                        help="μ-optimization iterations")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="μ-optimization learning rate")
    parser.add_argument("--tau-sweep", type=float, nargs="+", default=None,
                        help="Run a τ sweep and export all runs to JSON")
    parser.add_argument("--steps-sweep", type=int, nargs="+", default=None,
                        help="Run a step-count sweep and export all runs to JSON")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-conf", type=float, default=0.70)
    # ── Visualisation flags ──
    parser.add_argument("--viz", action="store_true",
                        help="Generate attribution heatmap plot")
    parser.add_argument("--viz-path", type=str,
                        default="attribution_heatmaps.png",
                        help="Output path for heatmap plot")
    parser.add_argument("--viz-fidelity", action="store_true",
                        help="Generate step-fidelity φ_k plot")
    # ── Insertion / Deletion ──
    parser.add_argument("--insdel", action="store_true",
                        help="Compute pixel-based insertion/deletion AUC")
    parser.add_argument("--insdel-steps", type=int, default=100,
                        help="Number of steps for ins/del evaluation")
    parser.add_argument("--viz-insdel", action="store_true",
                        help="Generate insertion/deletion curve plot")
    # ── Region-based Insertion / Deletion ──
    parser.add_argument("--region-insdel", action="store_true",
                        help="Compute region-based ins/del (SIC-style)")
    parser.add_argument("--viz-region-insdel", action="store_true",
                        help="Generate region-based ins/del curve plot")
    parser.add_argument("--patch-size", type=int, default=14,
                        help="Grid patch size for region ins/del")
    parser.add_argument("--no-slic", action="store_true",
                        help="Use grid patches instead of SLIC superpixels")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip", type=int, default=0)


    args = parser.parse_args()
    set_seed(args.seed)

    from utilss import (
        get_device, run_insertion_deletion, run_region_insertion_deletion,
        visualize_step_fidelity, visualize_insertion_deletion,
    )
    from lam import visualize_attributions

    device = get_device(force=args.device)
    if args.tau_sweep is not None or args.steps_sweep is not None:
        from lam import load_image_and_model
        print("Loading ResNet-50 and image...")
        model, x, baseline, info = load_image_and_model(
            device, args.min_conf, skip=args.skip)
        tau_values = args.tau_sweep if args.tau_sweep is not None else [args.tau]
        step_values = (
            args.steps_sweep if args.steps_sweep is not None else [args.steps])
        runs = []
        for steps in step_values:
            for tau in tau_values:
                print(f"\nSweep run: N = {steps}, τ = {tau}")
                runs.append(_run_loaded_experiment(
                    model, x, baseline, info, device,
                    N=steps, tau=tau, mu_iter=args.iters, lr=args.lr))
        out_path = args.json or "sweep_results.json"
        with open(out_path, "w") as f:
            json.dump({"runs": runs}, f, indent=2)
        print(f"\nSweep results → {out_path}")
        raise SystemExit(0)

    results, methods, model, x, baseline, info = run_experiment(
        N=args.steps, device=device, min_conf=args.min_conf,
        tau=args.tau, mu_iter=args.iters, lr=args.lr, skip=args.skip)

    # ── Insertion / Deletion ──
    insertion_deletion_results = None
    if args.insdel or args.viz_insdel:
        insertion_deletion_results = run_insertion_deletion(
            model, x, baseline, methods, n_steps=args.insdel_steps)
        _validate_insertion_deletion_export(
            insertion_deletion_results, methods, args.insdel_steps)
        results["insertion_deletion"] = insertion_deletion_results

    if args.region_insdel or args.viz_region_insdel:
        run_region_insertion_deletion(
            model, x, baseline, methods,
            patch_size=args.patch_size,
            use_slic=not args.no_slic)

    # ── JSON export ──
    if args.json:
        if args.insdel and insertion_deletion_results is None:
            raise RuntimeError("--insdel was enabled but no scores were computed")
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults → {args.json}")

    # ── Visualisation ──
    if args.viz:
        visualize_attributions(x, methods, info, save_path=args.viz_path,
                               delta_f=results["model_info"]["delta_f"])

    if args.viz_fidelity:
        fpath = args.viz_path.replace(".png", "_fidelity.png")
        visualize_step_fidelity(methods, save_path=fpath)

    if args.viz_insdel:
        ipath = args.viz_path.replace(".png", "_insdel.png")
        visualize_insertion_deletion(methods, save_path=ipath)

    if args.viz_region_insdel:
        rpath = args.viz_path.replace(".png", "_region_insdel.png")
        visualize_insertion_deletion(methods, save_path=rpath,
                                     use_region=True)
