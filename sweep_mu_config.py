#!/usr/bin/env python3
"""Sweep μ-Optimized IG hyperparameters on the top selected images."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

from batch_eval import mean_std, tau_tag
from lam import ClassLogitModel
from u_optimize import _validate_insertion_deletion_export, mu_optimized_ig
from utilss import get_device, run_insertion_deletion, set_seed


METHOD_NAME = "μ-Optimized"
METRIC_KEYS = [
    "Q",
    "CV2",
    "Var_nu",
    "mu_entropy",
    "mu_max",
    "mu_l2_sq",
    "mu_active_count",
    "weighted_residual_l1",
    "weighted_residual_l2",
    "insertion_auc",
    "deletion_auc",
    "insdel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep tau and steps for μ-Optimized IG on top selected images."
    )
    parser.add_argument("--selection-csv", "--selected-csv", dest="selection_csv",
                        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--tau-grid", type=float, nargs="+",
                        default=[0.001, 0.005, 0.01, 0.05, 0.1, 1.0])
    parser.add_argument("--steps-grid", type=int, nargs="+",
                        default=[16, 32, 64, 128])
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--insdel-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sweeps"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def read_top_images(selection_csv: Path, num_images: int) -> list[dict[str, Any]]:
    with selection_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows.sort(key=lambda row: int(row["rank"]))
    output = []
    for row in rows[:num_images]:
        image_path = row.get("copied_image_path") or row.get("image_path")
        if not image_path and row.get("image_name"):
            image_path = str(selection_csv.with_suffix("") / row["image_name"])
        output.append({
            "rank": int(row["rank"]),
            "image_path": image_path,
            "image_name": row.get("image_name") or Path(image_path).name,
            "resnet50_selection_score": float(row.get("top1_score") or row.get("score")),
        })
    return output


def load_resnet50(device: torch.device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


def load_image(path: Path, transform, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB")).contiguous().unsqueeze(0).to(device)


@torch.no_grad()
def top1_target(backbone: torch.nn.Module, x: torch.Tensor) -> tuple[int, float]:
    probs = F.softmax(backbone(x), dim=1)[0]
    score, pred = probs.max(dim=0)
    return int(pred.item()), float(score.item())


def method_metrics(method: Any, insdel_payload: dict[str, Any]) -> dict[str, Any]:
    payload = method.to_dict()
    metrics = {
        "Q": payload.get("Q"),
        "CV2": payload.get("CV2"),
        "Var_nu": payload.get("Var_nu"),
        "mu_entropy": payload.get("mu_entropy"),
        "mu_max": payload.get("mu_max"),
        "mu_l2_sq": payload.get("mu_l2_sq"),
        "active_count": payload.get("mu_active_count"),
        "mu_active_count": payload.get("mu_active_count"),
        "weighted_residual_l1": payload.get("weighted_residual_l1"),
        "weighted_residual_l2": payload.get("weighted_residual_l2"),
    }
    scores = insdel_payload["methods"][METHOD_NAME]
    metrics.update({
        "insertion_auc": scores["insertion_auc"],
        "deletion_auc": scores["deletion_auc"],
        "insdel": scores["insdel"],
    })
    return metrics


def aggregate_image_metrics(image_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[Any]] = defaultdict(list)
    for metrics in image_metrics:
        for key in METRIC_KEYS:
            values[key].append(metrics.get(key))
    aggregate: dict[str, Any] = {}
    rename = {
        "insertion_auc": "insertion_auc",
        "deletion_auc": "deletion_auc",
        "insdel": "insdel",
        "Q": "Q",
        "CV2": "CV2",
        "Var_nu": "Var_nu",
        "mu_entropy": "mu_entropy",
        "mu_max": "mu_max",
        "mu_l2_sq": "mu_l2_sq",
        "mu_active_count": "active_count",
        "weighted_residual_l1": "weighted_residual_l1",
        "weighted_residual_l2": "weighted_residual_l2",
    }
    for key, output_key in rename.items():
        mean, std = mean_std(values[key])
        aggregate[f"mean_{output_key}"] = mean
        aggregate[f"std_{output_key}"] = std
    return aggregate


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return (
        args.output_dir / f"sweep_results_seed{args.seed}.json",
        args.output_dir / f"sweep_summary_seed{args.seed}.csv",
        args.output_dir / f"best_config_seed{args.seed}.json",
    )


def write_outputs(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    selection_csv: Path,
    num_images: int,
) -> None:
    results_json, summary_csv, best_json = output_paths(args)
    best = max(
        (row for row in results if row.get("mean_insdel") is not None),
        key=lambda row: float(row["mean_insdel"]),
        default=None,
    )
    best_config = None
    if best is not None:
        best_config = {
            "tau": best["tau"],
            "steps": best["steps"],
            "mean_insdel": best["mean_insdel"],
        }

    payload = {
        "seed": args.seed,
        "num_images": num_images,
        "model": "resnet50",
        "selection_csv": str(selection_csv),
        "results": results,
        "best_config": best_config,
    }
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if results:
        with summary_csv.open("w", newline="", encoding="utf-8") as file:
            fieldnames = list(results[0])
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    if best_config is not None:
        best_json.write_text(json.dumps(best_config, indent=2), encoding="utf-8")


def run_config(
    entries: list[dict[str, Any]],
    backbone: torch.nn.Module,
    transform,
    device: torch.device,
    steps: int,
    tau: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_metrics = []
    failures = []
    for entry in tqdm(entries, desc=f"N={steps} tau={tau:g}"):
        try:
            x = load_image(Path(entry["image_path"]), transform, device)
            target_class, target_score = top1_target(backbone, x)
            scalar_model = ClassLogitModel(backbone, target_class).to(device).eval()
            baseline = torch.zeros_like(x)
            method = mu_optimized_ig(
                scalar_model, x, baseline, N=steps, tau=tau,
                n_iter=args.iters, lr=args.lr)
            insdel = run_insertion_deletion(
                scalar_model, x, baseline, [method], n_steps=args.insdel_steps)
            _validate_insertion_deletion_export(insdel, [method], args.insdel_steps)
            metrics = method_metrics(method, insdel)
            metrics.update({
                "rank": entry["rank"],
                "image_name": entry["image_name"],
                "target_class": target_class,
                "target_score": target_score,
            })
            image_metrics.append(metrics)
        except Exception as exc:
            if not args.skip_errors:
                raise
            failures.append({
                "rank": entry.get("rank"),
                "image_name": entry.get("image_name"),
                "error": str(exc),
            })

    aggregate = aggregate_image_metrics(image_metrics)
    return {
        "tau": tau,
        "steps": steps,
        "method": METHOD_NAME,
        "num_images_success": len(image_metrics),
        "num_images_failed": len(failures),
        "mean_insertion_auc": aggregate.get("mean_insertion_auc"),
        "std_insertion_auc": aggregate.get("std_insertion_auc"),
        "mean_deletion_auc": aggregate.get("mean_deletion_auc"),
        "std_deletion_auc": aggregate.get("std_deletion_auc"),
        "mean_insdel": aggregate.get("mean_insdel"),
        "std_insdel": aggregate.get("std_insdel"),
        "mean_Q": aggregate.get("mean_Q"),
        "std_Q": aggregate.get("std_Q"),
        "mean_CV2": aggregate.get("mean_CV2"),
        "std_CV2": aggregate.get("std_CV2"),
        "mean_Var_nu": aggregate.get("mean_Var_nu"),
        "std_Var_nu": aggregate.get("std_Var_nu"),
        "mean_mu_entropy": aggregate.get("mean_mu_entropy"),
        "std_mu_entropy": aggregate.get("std_mu_entropy"),
        "mean_mu_max": aggregate.get("mean_mu_max"),
        "std_mu_max": aggregate.get("std_mu_max"),
        "mean_mu_l2_sq": aggregate.get("mean_mu_l2_sq"),
        "std_mu_l2_sq": aggregate.get("std_mu_l2_sq"),
        "mean_active_count": aggregate.get("mean_active_count"),
        "std_active_count": aggregate.get("std_active_count"),
        "image_metrics": image_metrics,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(force=args.device)
    selection_csv = Path(args.selection_csv)
    entries = read_top_images(selection_csv, args.num_images)
    backbone, transform = load_resnet50(device)
    results: list[dict[str, Any]] = []
    print(f"Selection CSV: {selection_csv}")
    print(f"Images: {len(entries)}")

    for steps in args.steps_grid:
        for tau in args.tau_grid:
            row = run_config(entries, backbone, transform, device, steps, tau, args)
            results.append(row)
            write_outputs(args, results, selection_csv, len(entries))
            print(
                f"N={steps} tau={tau:g}: "
                f"mean_insdel={row.get('mean_insdel')} "
                f"success={row['num_images_success']} failed={row['num_images_failed']}"
            )

    write_outputs(args, results, selection_csv, len(entries))
    _, _, best_path = output_paths(args)
    print(f"Best config -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
