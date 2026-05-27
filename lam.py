"""
lam.py — Integrated Gradients attribution methods and experiment utilities
=========================================================================

The active attribution surface contains exactly three straight-line methods.
All use γ_k = baseline + (k/N)(x - baseline), gradients at k = 0,...,N-1,
and Δf_k = f(γ_{k+1}) - f(γ_k).

Derived quantities reported alongside include Var_ν(φ), CV²(φ), and Q.

Methods implemented:
    IG           — straight line, uniform μ
    IDG-PDF      — straight line, μ_k ∝ |Δf_k|
    μ-Optimised  — straight line, μ optimized for -Q + L2

Usage:
    python lam.py                        # single run
    python lam.py --viz --viz-fidelity   # with plots
    python lam.py --json results.json    # export

Requirements: torch >= 2.0, torchvision
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

from utilss import (
    get_device, AttributionResult, StepInfo, InsDelScores,
    compute_Var_nu, compute_CV2, compute_Q, compute_all_metrics,
    compute_residual_diagnostics, compute_weight_diagnostics,
    completeness_rescale,
    compute_insertion_deletion, run_insertion_deletion,
    compute_region_insertion_deletion, run_region_insertion_deletion,
    visualize_step_fidelity, visualize_insertion_deletion,
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  MODEL WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

class ClassLogitModel(nn.Module):
    """Wrap a classifier → scalar logit for a target class.  Shape: (B,)."""

    def __init__(self, backbone: nn.Module, target_class: int):
        super().__init__()
        self.backbone = backbone
        self.target_class = target_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[:, self.target_class]


# ═════════════════════════════════════════════════════════════════════════════
# §2  GRADIENT UTILITIES  (fused forward + backward)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_scalar(model: nn.Module, x: torch.Tensor) -> float:
    return float(model(x).squeeze())


@torch.no_grad()
def _forward_batch(model: nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    """f(x) for a batch.  Returns (B,) tensor on same device."""
    return model(x_batch)


def _forward_and_gradient(model: nn.Module, x: torch.Tensor
                          ) -> tuple[float, torch.Tensor]:
    """f(x) and ∇f(x) in ONE backward pass."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        out = model(x_in).sum()
        f_val = float(out)
        out.backward()
    return f_val, x_in.grad.detach()


def _forward_and_gradient_batch(model: nn.Module, x_batch: torch.Tensor
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched f(x) and ∇_x f(x).

    Args:
        x_batch: (B, C, H, W)

    Returns:
        f_vals: (B,) tensor of scalar outputs
        grads:  (B, C, H, W) tensor of per-sample gradients

    Uses torch.vmap-style trick: we sum all outputs and backward once,
    but because each output depends only on its own input row the
    cross-gradients are zero and x_in.grad gives per-sample gradients.
    """
    B = x_batch.shape[0]
    with torch.enable_grad():
        x_in = x_batch.detach().clone().requires_grad_(True)
        model.zero_grad()
        # model returns (B,) — one scalar per sample
        outs = model(x_in)          # (B,)
        f_vals = outs.detach()      # (B,)
        outs.sum().backward()
    return f_vals, x_in.grad.detach()


def _gradient(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """∇f(x) only (when f value already known)."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        model(x_in).sum().backward()
    return x_in.grad.detach()


def _gradient_batch(model: nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    """Batched ∇f(x).  Returns (B, C, H, W)."""
    with torch.enable_grad():
        x_in = x_batch.detach().clone().requires_grad_(True)
        model.zero_grad()
        model(x_in).sum().backward()
    return x_in.grad.detach()


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


# ═════════════════════════════════════════════════════════════════════════════
# §3  STEP DIAGNOSTICS BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _rescale(attr: torch.Tensor, target: float) -> torch.Tensor:
    return completeness_rescale(attr, target)


def _build_steps(d_list, df_list, f_vals, gnorms, mu, N) -> list[StepInfo]:
    steps = []
    for k in range(N):
        dk, dfk = d_list[k], df_list[k]
        rk = dfk - dk
        phik = dk / dfk if abs(dfk) > 1e-12 else 1.0
        steps.append(StepInfo(
            t=k / N, f=f_vals[k], d_k=dk, delta_f_k=dfk,
            r_k=rk, phi_k=phik, grad_norm=gnorms[k], mu_k=float(mu[k]),
        ))
    return steps


def _pack_result(name, attr, d_list, df_list, f_vals, gnorms, mu, N,
                 t0, Q_history=None, objective_history=None,
                 optimizer_info=None) -> AttributionResult:
    """Build AttributionResult with all three metrics in one pass."""
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


# ═════════════════════════════════════════════════════════════════════════════
# §4  STRAIGHT-LINE PASS  (shared by IG, IDG-PDF, μ-Optimised)
#
#     FIX 1: batch all N interpolation points into ONE forward + ONE backward.
#     Old cost:  N sequential forward+backward = 2N model calls
#     New cost:  1 batched forward+backward     = 2 model calls  (+ 2 scalar)
# ═════════════════════════════════════════════════════════════════════════════

def _straight_line_pass(model: nn.Module, x: torch.Tensor,
                        baseline: torch.Tensor, N: int,
                        fwd_batch_size: int = 0):
    """
    Evaluate f and ∇f at N uniformly-spaced points along the straight line.

    FIX 1: All N points are stacked into a single (N, C, H, W) batch and
    processed in one forward+backward call (or chunked if fwd_batch_size > 0
    to limit GPU memory).

    Returns: (delta_x, target, grads, d_list, df_list, f_vals, gnorms)
        grads   : list of N gradient tensors  (each (1, C, H, W))
        d_list  : list of N floats (d_k = ∇f(γ_k)·((x - baseline)/N))
        df_list : list of N floats (Δf_k)
        f_vals  : list of N+1 floats (f at γ_0, ..., γ_N)
        gnorms  : list of N floats (‖∇f‖)
    """
    delta_x = x - baseline
    step = delta_x / N
    # Endpoints — scalar, cheap
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    # Build batch of N interpolation points: γ_k = baseline + (k/N) * delta_x
    # alphas shape (N, 1, 1, 1) for broadcasting
    alphas = torch.arange(N, device=x.device, dtype=x.dtype).view(N, 1, 1, 1) / N
    gamma_batch = baseline + alphas * delta_x       # (N, C, H, W)

    # ── Batched forward + backward ──
    if fwd_batch_size <= 0 or fwd_batch_size >= N:
        # Single shot
        f_batch, grad_batch = _forward_and_gradient_batch(model, gamma_batch)
    else:
        # Chunked to limit VRAM
        f_chunks, g_chunks = [], []
        for i0 in range(0, N, fwd_batch_size):
            i1 = min(i0 + fwd_batch_size, N)
            fb, gb = _forward_and_gradient_batch(model, gamma_batch[i0:i1])
            f_chunks.append(fb)
            g_chunks.append(gb)
        f_batch = torch.cat(f_chunks, dim=0)        # (N,)
        grad_batch = torch.cat(g_chunks, dim=0)      # (N, C, H, W)

    # f_vals layout: [f(γ_0), f(γ_1), ..., f(γ_N)]
    # Gradients are evaluated at γ_k for k = 0,...,N-1 and
    # Δf_k = f(γ_{k+1}) - f(γ_k).
    f_vals = f_batch.tolist() + [f_x]

    # d_k = ∇f(γ_k) · ((x - baseline) / N)
    d_tensor = (grad_batch * step).view(N, -1).sum(dim=1)   # (N,)
    d_list = d_tensor.tolist()

    # Δf_k = f_vals[k+1] - f_vals[k]
    df_list = [f_vals[k + 1] - f_vals[k] for k in range(N)]

    # grad norms
    gnorms = grad_batch.view(N, -1).norm(dim=1).tolist()     # N floats

    # grads as list of (1, C, H, W) — clone to avoid shared-memory bugs
    grads = [grad_batch[k:k+1].clone() for k in range(N)]

    return delta_x, target, grads, d_list, df_list, f_vals, gnorms


# ═════════════════════════════════════════════════════════════════════════════
# §5  STANDARD IG
# ═════════════════════════════════════════════════════════════════════════════

def standard_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
                N: int = 50) -> AttributionResult:
    """Standard IG (Sundararajan et al., 2017).  No optimisation."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    # grads[k] is (1, C, H, W) — stack and mean
    grad_sum = torch.cat(grads, dim=0).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = completeness_rescale(delta_x * grad_sum / N, target)
    mu = torch.full((N,), 1.0 / N, device=x.device)

    return _pack_result("IG", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §6  IDG-PDF
# ═════════════════════════════════════════════════════════════════════════════

def idg_pdf(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
         N: int = 50) -> AttributionResult:
    """IDG-PDF: output-change weighted IG with μ_k ∝ |Δf_k|."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    df_arr = torch.tensor(df_list, device=x.device)
    weights = df_arr.abs()
    w_sum = weights.sum()
    mu = weights / w_sum if w_sum > 1e-12 else torch.full((N,), 1.0/N, device=x.device)

    # Weighted gradient sum — grads[k] is (1, C, H, W)
    grad_stack = torch.cat(grads, dim=0)               # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)                         # (N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = completeness_rescale(delta_x * wg, target)

    return _pack_result("IDG-PDF", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)

# ═════════════════════════════════════════════════════════════════════════════
# §7  μ-OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu(d: torch.Tensor, delta_f: torch.Tensor,
                tau: float = 0.01, n_iter: int = 200,
                lr: float = 0.05, log_every: int = 20,
                return_history: bool = False):
    """
    Find μ minimizing -Q(μ) + (τ/2)||μ||² with projected gradient descent.
    """
    device = d.device
    N = d.shape[0]
    mu = torch.full((N,), 1.0 / N, device=device, dtype=d.dtype)

    a = (d * delta_f).detach()
    b = (d ** 2).detach()
    c = (delta_f ** 2).detach()
    q_history = []
    objective_history = []
    n_done = 0

    def _objective(mu_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        P = (mu_vec * a).sum()
        D = (mu_vec * b).sum()
        Fv = (mu_vec * c).sum()
        if D.item() <= 1e-15 or Fv.item() <= 1e-15:
            q_val = torch.zeros((), device=device, dtype=d.dtype)
        else:
            q_val = (P ** 2 / (D * Fv)).clamp(0.0, 1.0)
        obj_val = -q_val + (tau / 2.0) * (mu_vec ** 2).sum()
        return q_val, obj_val

    def _record(iteration: int) -> None:
        q_val, obj_val = _objective(mu)
        q_history.append({"iteration": iteration, "Q": float(q_val)})
        objective_history.append(
            {"iteration": iteration, "objective": float(obj_val)})

    for iteration in range(1, n_iter + 1):
        P = (mu * a).sum()
        D = (mu * b).sum()
        Fv = (mu * c).sum()
        if D.item() <= 1e-15 or Fv.item() <= 1e-15:
            break
        grad_q = (P / (D * Fv)) * (2.0 * a - (P / D) * b - (P / Fv) * c)
        grad = -grad_q + tau * mu
        mu = project_simplex(mu - lr * grad)
        n_done = iteration
        if return_history and log_every > 0 and iteration % log_every == 0:
            _record(iteration)

    if return_history:
        if not q_history or q_history[-1]["iteration"] != n_done:
            _record(n_done)
        _, final_objective = _objective(mu)
        history = {
            "Q_history": q_history,
            "objective_history": objective_history,
            "final_objective": float(final_objective),
            "n_iters": int(n_done),
            "tau": float(tau),
            "learning_rate": float(lr),
        }
        return mu.detach(), history
    return mu.detach()


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {μ >= 0, Σμ = 1}."""
    if v.ndim != 1:
        raise ValueError("project_simplex expects a 1D tensor")
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if cond.sum().item() == 0:
        return torch.full_like(v, 1.0 / n)
    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    theta = cssv[rho] / (rho.to(v.dtype) + 1.0)
    return torch.clamp(v - theta, min=0.0)


# ═════════════════════════════════════════════════════════════════════════════
# §8  μ-OPTIMISED IG
# ═════════════════════════════════════════════════════════════════════════════

def mu_optimized_ig(model: nn.Module, x: torch.Tensor,
                    baseline: torch.Tensor, N: int = 50,
                    tau: float = 0.005, n_iter: int = 300,
                    lr: float = 0.05, log_every: int = 20,
                    ) -> AttributionResult:
    """Straight line + optimal μ.  Cost = standard IG + O(N) arithmetic."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    d_arr = torch.tensor(d_list, device=x.device)
    df_arr = torch.tensor(df_list, device=x.device)
    mu, history = optimize_mu(
        d_arr, df_arr, tau=tau, n_iter=n_iter, lr=lr,
        log_every=log_every, return_history=True)

    # Weighted gradient sum
    grad_stack = torch.cat(grads, dim=0)               # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = _rescale(delta_x * wg, target)

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
# §9  IMAGE LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_image_and_model(device: torch.device, min_conf: float = 0.70, skip=0):
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    x, pc, cf = None, None, None
    source = "none"

    # Try local images
    for sample_dir in ["./sample_imagenet1k", "../sample_imagenet1k",
                       os.path.expanduser("~/sample_imagenet1k")]:
        if not os.path.isdir(sample_dir):
            continue
        try:
            from PIL import Image
            import random
            jpegs = sorted([f for f in os.listdir(sample_dir)
                            if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            random.shuffle(jpegs)
            print(f"Found {sample_dir} ({len(jpegs)} images)")
            cskip = 0
            for fname in jpegs:
                try:
                    img = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
                except Exception:
                    continue
                xc = tf(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(backbone(xc), dim=-1)
                    c, pr = p[0].max(0)
                if c.item() >= min_conf:
                    cskip += 1
                    if skip > 0 and cskip <= skip:
                        continue
                    x, pc, cf = xc, pr.item(), c.item()
                    source = f"{sample_dir}/{fname}"
                    print(f"  ✓ {fname} → class={pc}, conf={cf:.4f}")
                    break
        except Exception as e:
            print(f"  Error: {e}")
        if x is not None:
            break

    # Fallback: CIFAR-10
    if x is None:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406],
                                         [0.229,0.224,0.225])])
            ds = CIFAR10("./data", train=False, download=True, transform=ctf)
            for i in range(500):
                im, _ = ds[i]
                xc = im.unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(backbone(xc), dim=-1)
                    c, pr = p[0].max(0)
                if c.item() >= min_conf:
                    x, pc, cf = xc, pr.item(), c.item()
                    source = f"CIFAR-10 idx={i}"
                    break
        except Exception:
            pass

    # Fallback: synthetic
    if x is None:
        print("Using synthetic image fallback")
        m = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        s = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
        torch.manual_seed(42)
        raw = (torch.randn(1,3,224,224, device=device)*0.2+0.5).clamp(0,1)
        x = (raw - m) / s
        with torch.no_grad():
            p = F.softmax(backbone(x), dim=-1)
            c, pr = p[0].max(0)
            pc, cf = pr.item(), c.item()
        source = "synthetic"

    model = ClassLogitModel(backbone, target_class=pc).to(device).eval()
    baseline = torch.zeros_like(x)

    # Extract human-readable class name from filename if possible
    # ImageNet format: nXXXXXXXX_class_name.JPEG
    class_name = None
    if "/" in source:
        fname = source.rsplit("/", 1)[-1]
        name_part = fname.rsplit(".", 1)[0]        # strip extension
        parts = name_part.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("n") and parts[0][1:].isdigit():
            class_name = parts[1].replace("_", " ")

    info = {"source": source, "target_class": pc, "confidence": cf,
            "model": "ResNet-50 (ImageNet pretrained)",
            "class_name": class_name}
    return model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §10  HEATMAP VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def visualize_attributions(x, methods, info, save_path="attribution_heatmaps.png",
                           delta_f=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(x.device)
    img = ((x*std+mean).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype("uint8")
    img_dark = (img.astype(float)*0.4).astype("uint8")

    cmap = LinearSegmentedColormap.from_list("heat", [
        (0,(0,0,0,0)), (0.3,(0.97,0.45,0.02,0.4)),
        (0.6,(0.97,0.71,0.22,0.7)), (1,(1,1,1,1))])
    colors = {"IG":"#6B7280","IDG-PDF":"#8B5CF6",
              "μ-Optimized":"#F59E0B"}

    n = len(methods)
    fig, axes = plt.subplots(2, n+1, figsize=(3.6*(n+1), 7.5), facecolor="#0D0D0D")

    # Build title from class name or class index
    class_name = info.get("class_name")
    class_id = info["target_class"]
    if class_name:
        label = f'"{class_name}" (class {class_id})'
    else:
        label = f"class {class_id}"
    fig.suptitle(
        f"{label}  ·  conf {info['confidence']:.1%}  ·  Δf = {delta_f:.2f}",
        color="#E8E4DF", fontsize=12, fontfamily="monospace",
        fontweight="bold", y=0.98,
    )

    axes[0,0].imshow(img); axes[0,0].set_title("Original", color="#E8E4DF",
        fontsize=10, fontfamily="monospace"); axes[0,0].axis("off")

    for i, m in enumerate(methods):
        sal = m.attributions[0].abs().sum(0).cpu().numpy()
        vmax = max(np.percentile(sal, 99), 1e-12)
        sal = (sal/vmax).clip(0,1)
        ax = axes[0, i+1]
        ax.imshow(img_dark); ax.imshow(sal, cmap=cmap, vmin=0, vmax=1, alpha=0.85)
        c = colors.get(m.name, "#F7B538")
        ax.set_title(f"{m.name}\n𝒬={m.Q:.4f}  Var={m.Var_nu:.4f}",
                     color=c, fontsize=9, fontfamily="monospace", linespacing=1.4)
        ax.axis("off")

    # Bottom row: Q bar chart + signed heatmaps
    ax_bar = axes[1, 0]; ax_bar.set_facecolor("#0D0D0D")
    qs = [m.Q for m in methods]; names = [m.name for m in methods]
    cs = [colors.get(n, "#F7B538") for n in names]
    bars = ax_bar.barh(range(n), qs, color=cs, height=0.6)
    for bar, q in zip(bars, qs):
        ax_bar.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                    f"{q:.4f}", va="center", color="#E8E4DF", fontsize=8,
                    fontfamily="monospace")
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(names, fontsize=8, fontfamily="monospace", color="#E8E4DF")
    ax_bar.set_xlim(0, 1.15); ax_bar.invert_yaxis()
    ax_bar.set_title("𝒬 Score", color="#E8E4DF", fontsize=10, fontfamily="monospace")
    ax_bar.tick_params(colors="#888", labelsize=7)
    for sp in ax_bar.spines.values(): sp.set_color("#333")

    cmap_div = LinearSegmentedColormap.from_list("div", [
        (0,(0.15,0.35,0.85,0.9)), (0.5,(0,0,0,0)), (1,(0.95,0.2,0.1,0.9))])
    for i, m in enumerate(methods):
        sal = m.attributions[0].sum(0).cpu().numpy()
        vmax = max(np.percentile(np.abs(sal), 99), 1e-12)
        ax = axes[1, i+1]; ax.imshow(img_dark)
        ax.imshow((sal/vmax).clip(-1,1), cmap=cmap_div, vmin=-1, vmax=1, alpha=0.85)
        ax.set_title(f"Signed · {m.name}", color=colors.get(m.name,"#F7B538"),
                     fontsize=9, fontfamily="monospace"); ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor="#0D0D0D",
                bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"✓ Heatmap → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# §11  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def _run_methods(model, x, baseline, N, tau, mu_iter, lr):
    return [
        standard_ig(model, x, baseline, N),
        idg_pdf(model, x, baseline, N),
        mu_optimized_ig(
            model, x, baseline, N, tau=tau, n_iter=mu_iter, lr=lr),
    ]


def _result_payload(methods, info, f_x, f_bl, delta_f, N, device,
                    tau, mu_iter, lr):
    return {
        "config": {"N": N, "tau": tau, "iters": mu_iter, "lr": lr},
        "image_info": info,
        "model_info": {"f_x": f_x, "f_baseline": f_bl,
                       "delta_f": delta_f, "N": N, "device": str(device)},
        "methods": {m.name: m.to_dict() for m in methods},
    }


def run_experiment(N=50, device=None, min_conf=0.70, tau=0.005,
                   mu_iter=300, lr=0.05, skip=0):
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
    print(f"{'Method':<16} {'Var_ν':>10} {'CV²':>8} {'𝒬':>8} {'Time':>8}")
    print("─" * 56)

    methods = _run_methods(model, x, baseline, N, tau, mu_iter, lr)

    for m in methods:
        print(f"{m.name:<16} {m.Var_nu:>10.6f} {m.CV2:>8.4f} "
              f"{m.Q:>8.4f} {m.elapsed_s:>7.1f}s")

    results = _result_payload(
        methods, info, f_x, f_bl, delta_f, N, device, tau, mu_iter, lr)
    return results, methods, model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §12  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="μ-Optimized IG: IG, IDG-PDF, and μ-Optimized")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tau-sweep", type=float, nargs="+", default=None)
    parser.add_argument("--steps-sweep", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-conf", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-path", type=str, default="attribution_heatmaps.png")
    parser.add_argument("--viz-fidelity", action="store_true")
    parser.add_argument("--insdel", action="store_true")
    parser.add_argument("--insdel-steps", type=int, default=100)
    parser.add_argument("--viz-insdel", action="store_true")
    parser.add_argument("--region-insdel", action="store_true",
                        help="Compute region-based insertion/deletion (SIC-style)")
    parser.add_argument("--viz-region-insdel", action="store_true",
                        help="Generate region-based ins/del curve plot")
    parser.add_argument("--patch-size", type=int, default=14,
                        help="Grid patch size for region ins/del (default: 14)")
    parser.add_argument("--no-slic", action="store_true",
                        help="Use grid patches instead of SLIC superpixels")
    args = parser.parse_args()

    from utilss import set_seed
    set_seed(args.seed)
    device = get_device(force=args.device)
    if args.tau_sweep is not None or args.steps_sweep is not None:
        print("Loading ResNet-50 and image...")
        model, x, baseline, info = load_image_and_model(
            device, args.min_conf, skip=args.skip)
        f_x = _forward_scalar(model, x)
        f_bl = _forward_scalar(model, baseline)
        delta_f = f_x - f_bl
        tau_values = args.tau_sweep if args.tau_sweep is not None else [args.tau]
        step_values = (
            args.steps_sweep if args.steps_sweep is not None else [args.steps])
        runs = []
        for steps in step_values:
            for tau in tau_values:
                print(f"\nSweep run: N = {steps}, τ = {tau}")
                methods = _run_methods(
                    model, x, baseline, steps, tau, args.iters, args.lr)
                runs.append(_result_payload(
                    methods, info, f_x, f_bl, delta_f, steps, device,
                    tau, args.iters, args.lr))
        out_path = args.json or "sweep_results.json"
        with open(out_path, "w") as f:
            json.dump({"runs": runs}, f, indent=2)
        print(f"\nSweep results → {out_path}")
        raise SystemExit(0)

    results, methods, model, x, baseline, info = run_experiment(
        N=args.steps, device=device, min_conf=args.min_conf,
        tau=args.tau, mu_iter=args.iters, lr=args.lr, skip=args.skip)

    if args.insdel or args.viz_insdel:
        run_insertion_deletion(model, x, baseline, methods,
                               n_steps=args.insdel_steps)

    if args.region_insdel or args.viz_region_insdel:
        run_region_insertion_deletion(
            model, x, baseline, methods,
            patch_size=args.patch_size,
            use_slic=not args.no_slic)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults → {args.json}")

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
