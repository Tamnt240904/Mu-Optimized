#!/usr/bin/env python3
"""Batch attribution and insertion/deletion evaluation for local images."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from lam import ClassLogitModel
from u_optimize import _validate_insertion_deletion_export, run_all_methods
from utilss import get_device, run_insertion_deletion, set_seed


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
METHOD_NAMES = ["IG", "IDG-PDF", "μ-Optimized"]
METHOD_SUMMARY_KEYS = [
    "Q",
    "CV2",
    "Var_nu",
    "mu_l2_sq",
    "mu_entropy",
    "mu_max",
    "mu_min",
    "mu_active_count",
    "mu_sum",
    "weighted_residual_l1",
    "weighted_residual_l2",
    "elapsed_s",
]
AGGREGATE_KEYS = ["Q", "insertion_auc", "deletion_auc", "insdel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IG, IDG-PDF, and μ-Optimized IG on multiple images."
    )
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--image-dir", type=str, default="sample_imagenet1k")
    parser.add_argument("--output-json", type=str,
                        default="results/batch_auc_N64_tau001.json")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--insdel", action="store_true")
    parser.add_argument("--insdel-steps", type=int, default=50)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def imagenet_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def list_images(image_dir: Path, num_images: int) -> list[Path]:
    if num_images <= 0:
        raise ValueError(f"--num-images must be positive, got {num_images}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"no image files found in: {image_dir}")
    return images[:num_images]


def load_backbone(device: torch.device) -> torch.nn.Module:
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


def class_name_from_path(path: Path) -> str | None:
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("n") and parts[0][1:].isdigit():
        return parts[1].replace("_", " ")
    return None


def load_image_tensor(path: Path, transform: T.Compose,
                      device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


@torch.no_grad()
def select_target(backbone: torch.nn.Module, x: torch.Tensor,
                  target_class: int | None) -> tuple[int, float]:
    probs = F.softmax(backbone(x), dim=-1)[0]
    if target_class is None:
        confidence, predicted = probs.max(0)
        return int(predicted.item()), float(confidence.item())
    if target_class < 0 or target_class >= probs.numel():
        raise ValueError(
            f"--target-class must be in [0, {probs.numel() - 1}], got {target_class}"
        )
    return target_class, float(probs[target_class].item())


def method_summary(method: Any) -> dict[str, Any]:
    payload = method.to_dict()
    return {key: payload.get(key) for key in METHOD_SUMMARY_KEYS}


def validate_successful_image(image_result: dict[str, Any], insdel_steps: int,
                              require_insdel: bool) -> None:
    methods = image_result.get("methods", {})
    if set(methods) != set(METHOD_NAMES):
        missing = sorted(set(METHOD_NAMES) - set(methods))
        extra = sorted(set(methods) - set(METHOD_NAMES))
        raise ValueError(
            f"{image_result.get('image_name')}: method mismatch; "
            f"missing={missing}, extra={extra}"
        )

    for method_name, metrics in methods.items():
        q = metrics.get("Q")
        if not isinstance(q, (int, float)) or not (-1e-7 <= q <= 1.0 + 1e-7):
            raise ValueError(f"{image_result.get('image_name')} {method_name}: invalid Q")
        mu_sum = metrics.get("mu_sum")
        if not isinstance(mu_sum, (int, float)) or abs(mu_sum - 1.0) >= 1e-5:
            raise ValueError(
                f"{image_result.get('image_name')} {method_name}: invalid mu_sum"
            )

    if require_insdel:
        insertion_deletion = image_result.get("insertion_deletion")
        if not isinstance(insertion_deletion, dict):
            raise ValueError(
                f"{image_result.get('image_name')}: missing insertion_deletion"
            )
        for method_name, metrics in insertion_deletion.get("methods", {}).items():
            insertion_curve = metrics.get("insertion_curve")
            deletion_curve = metrics.get("deletion_curve")
            if not isinstance(insertion_curve, list):
                raise ValueError(
                    f"{image_result.get('image_name')} {method_name}: "
                    "insertion_curve is not a list"
                )
            if not isinstance(deletion_curve, list):
                raise ValueError(
                    f"{image_result.get('image_name')} {method_name}: "
                    "deletion_curve is not a list"
                )
            if len(insertion_curve) != insdel_steps + 1:
                raise ValueError(
                    f"{image_result.get('image_name')} {method_name}: "
                    "invalid insertion_curve length"
                )
            if len(deletion_curve) != insdel_steps + 1:
                raise ValueError(
                    f"{image_result.get('image_name')} {method_name}: "
                    "invalid deletion_curve length"
                )
            if not math.isfinite(metrics.get("insertion_auc")):
                raise ValueError(
                    f"{image_result.get('image_name')} {method_name}: "
                    "insertion_auc is not finite"
                )
            if not math.isfinite(metrics.get("deletion_auc")):
                raise ValueError(
                    f"{image_result.get('image_name')} {method_name}: "
                    "deletion_auc is not finite"
                )


def run_one_image(
    image_path: Path,
    backbone: torch.nn.Module,
    transform: T.Compose,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x = load_image_tensor(image_path, transform, device)
    target_class, confidence = select_target(backbone, x, args.target_class)
    model = ClassLogitModel(backbone, target_class=target_class).to(device).eval()
    baseline = torch.zeros_like(x)
    methods = run_all_methods(
        model,
        x,
        baseline,
        N=args.steps,
        tau=args.tau,
        mu_iter=args.iters,
        lr=args.lr,
    )

    image_result: dict[str, Any] = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "target_class": target_class,
        "confidence": confidence,
        "class_name": class_name_from_path(image_path),
        "success": True,
        "methods": {method.name: method_summary(method) for method in methods},
    }

    if args.insdel:
        insertion_deletion = run_insertion_deletion(
            model, x, baseline, methods, n_steps=args.insdel_steps)
        _validate_insertion_deletion_export(
            insertion_deletion, methods, args.insdel_steps)
        image_result["insertion_deletion"] = insertion_deletion

    validate_successful_image(image_result, args.insdel_steps, args.insdel)
    return image_result


def finite_values(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(value):
            output.append(float(value))
    return output


def mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    vals = finite_values(values)
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    variance = sum((value - mean) ** 2 for value in vals) / len(vals)
    return mean, math.sqrt(variance)


def compute_aggregate(images: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for image in images:
        if not image.get("success"):
            continue
        for method_name, metrics in image.get("methods", {}).items():
            values[method_name]["Q"].append(metrics.get("Q"))
        insertion_deletion = image.get("insertion_deletion", {})
        for method_name, metrics in insertion_deletion.get("methods", {}).items():
            values[method_name]["insertion_auc"].append(metrics.get("insertion_auc"))
            values[method_name]["deletion_auc"].append(metrics.get("deletion_auc"))
            values[method_name]["insdel"].append(metrics.get("insdel"))

    aggregate = {"methods": {}}
    for method_name in METHOD_NAMES:
        method_agg = {}
        for key in AGGREGATE_KEYS:
            mean, std = mean_std(values[method_name][key])
            method_agg[f"{key}_mean"] = mean
            method_agg[f"{key}_std"] = std
        aggregate["methods"][method_name] = method_agg
    return aggregate


def fmt_mean_std(mean: Any, std: Any) -> str:
    if mean is None or std is None:
        return "n/a"
    return f"{mean:.4f}±{std:.4f}"


def print_summary_table(aggregate: dict[str, Any]) -> None:
    print("\nBatch summary")
    print(
        f"{'Method':<14} {'Q mean±std':>18} "
        f"{'Insertion AUC mean±std':>28} "
        f"{'Deletion AUC mean±std':>27} {'Ins-Del mean±std':>20}"
    )
    print("-" * 111)
    for method_name in METHOD_NAMES:
        metrics = aggregate["methods"].get(method_name, {})
        print(
            f"{method_name:<14} "
            f"{fmt_mean_std(metrics.get('Q_mean'), metrics.get('Q_std')):>18} "
            f"{fmt_mean_std(metrics.get('insertion_auc_mean'), metrics.get('insertion_auc_std')):>28} "
            f"{fmt_mean_std(metrics.get('deletion_auc_mean'), metrics.get('deletion_auc_std')):>27} "
            f"{fmt_mean_std(metrics.get('insdel_mean'), metrics.get('insdel_std')):>20}"
        )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(force=args.device)
    image_paths = list_images(Path(args.image_dir), args.num_images)

    backbone = load_backbone(device)
    transform = imagenet_transform()
    images: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        try:
            images.append(run_one_image(image_path, backbone, transform, device, args))
        except Exception as exc:
            if not args.skip_errors:
                raise
            print(f"  ERROR: {exc}")
            images.append({
                "image_path": str(image_path),
                "image_name": image_path.name,
                "success": False,
                "error": str(exc),
            })

    aggregate = compute_aggregate(images)
    output = {
        "config": {
            "num_images": len(image_paths),
            "image_dir": args.image_dir,
            "steps": args.steps,
            "tau": args.tau,
            "iters": args.iters,
            "lr": args.lr,
            "target_class": args.target_class,
            "insdel": args.insdel,
            "insdel_steps": args.insdel_steps,
            "seed": args.seed,
            "skip_errors": args.skip_errors,
        },
        "images": images,
        "aggregate": aggregate,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print_summary_table(aggregate)
    print(f"\nBatch results -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
