#!/usr/bin/env python3
"""Batch attribution and insertion/deletion evaluation for selected images."""

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
AGGREGATE_KEYS = [
    "Q",
    "CV2",
    "Var_nu",
    "mu_l2_sq",
    "mu_entropy",
    "mu_max",
    "weighted_residual_l1",
    "weighted_residual_l2",
    "insertion_auc",
    "deletion_auc",
    "insdel",
]
MODEL_LOADERS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": {models.resnet101, models.ResNet50_Weights.DEFAULT},
    "vgg19": {models.vgg19, models.VGG16_Weights.DEFAULT},
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IG, IDG-PDF, and μ-Optimized IG on multiple images."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to evaluate. If omitted, evaluate all selected rows/images.",
    )
    parser.add_argument("--selected-csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str,
                        default="generated_imagenet/imagenet_resnet50_correct_1000")
    parser.add_argument("--model-name", choices=sorted(MODEL_LOADERS),
                        default="resnet50")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--insdel", action="store_true")
    parser.add_argument("--insdel-steps", type=int, default=50)
    parser.add_argument("--auc-score", choices=["logit", "confidence"], default="logit")
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--target-from-csv", choices=["resnet50_pred"], default=None)
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


def tau_tag(tau: float) -> str:
    return f"{tau:g}".replace(".", "p").replace("-", "m")


def default_output_path(args: argparse.Namespace) -> Path:
    return Path(
        "results"
    ) / f"full_eval_seed{args.seed}_{args.model_name}_N{args.steps}_tau{tau_tag(args.tau)}.json"


def load_backbone(model_name: str, device: torch.device):
    builder, weights = MODEL_LOADERS[model_name]
    backbone = builder(weights=weights).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    transform = weights.transforms() if hasattr(weights, "transforms") else imagenet_transform()
    categories = weights.meta.get("categories", [])
    return backbone, transform, categories


def list_images(image_dir: Path, num_images: int | None) -> list[dict[str, Any]]:
    if num_images is not None and num_images <= 0:
        raise ValueError(f"--num-images must be positive, got {num_images}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"no image files found in: {image_dir}")
    selected_images = images if num_images is None else images[:num_images]
    return [
        {
            "rank": index,
            "image_path": str(path),
            "copied_image_path": str(path),
            "image_name": path.name,
            "resnet50_selection_score": None,
            "resnet50_predicted_label_idx": None,
        }
        for index, path in enumerate(selected_images, start=1)
    ]


def read_selected_csv(csv_path: Path, num_images: int | None) -> list[dict[str, Any]]:
    if num_images is not None and num_images <= 0:
        raise ValueError(f"--num-images must be positive, got {num_images}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"selected CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows.sort(key=lambda row: parse_int(row.get("rank")) or 10**9)
    selected_rows = rows if num_images is None else rows[:num_images]
    selected = []
    for fallback_rank, row in enumerate(selected_rows, start=1):
        image_path = row.get("copied_image_path") or row.get("image_path")
        if not image_path and row.get("image_name"):
            sibling_dir = csv_path.with_suffix("")
            image_path = str(sibling_dir / row["image_name"])
        if not image_path:
            raise ValueError(f"missing image path in {csv_path}")
        top1_score = parse_float(row.get("top1_score") or row.get("score"))
        entry = dict(row)
        entry.update({
            "rank": parse_int(row.get("rank")) or fallback_rank,
            "image_path": image_path,
            "copied_image_path": row.get("copied_image_path", image_path),
            "image_name": row.get("image_name") or Path(image_path).name,
            "resnet50_selection_score": top1_score,
            "selection_top1_score": top1_score,
            "resnet50_predicted_label_idx": parse_int(
                row.get("predicted_label_idx") or row.get("resnet50_predicted_label_idx")
            ),
        })
        for key in (
            "correct_index",
            "source_order",
            "ground_truth_label_idx",
            "predicted_label_idx",
        ):
            if key in row:
                entry[key] = parse_int(row.get(key))
        for key in ("top1_score",):
            if key in row:
                entry[key] = parse_float(row.get(key))
        selected.append(entry)
    return selected


def load_image_tensor(path: Path, transform: T.Compose,
                      device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB")).contiguous().unsqueeze(0).to(device)


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


@torch.no_grad()
def select_target(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    target_class: int | None,
) -> tuple[int, float]:
    probs = F.softmax(backbone(x), dim=-1)[0]
    if target_class is None:
        confidence, predicted = probs.max(0)
        return int(predicted.item()), float(confidence.item())
    if target_class < 0 or target_class >= probs.numel():
        raise ValueError(
            f"target class must be in [0, {probs.numel() - 1}], got {target_class}"
        )
    return target_class, float(probs[target_class].item())


def method_summary(method: Any) -> dict[str, Any]:
    payload = method.to_dict()
    return {key: payload.get(key) for key in METHOD_SUMMARY_KEYS}


def validate_successful_image(image_result: dict[str, Any], insdel_steps: int,
                              require_insdel: bool) -> None:
    methods = image_result.get("methods", {})
    if set(methods) != set(METHOD_NAMES):
        raise ValueError(f"{image_result.get('image_name')}: method mismatch")
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
            raise ValueError(f"{image_result.get('image_name')}: missing insdel")
        for method_name, metrics in insertion_deletion.get("methods", {}).items():
            if len(metrics.get("insertion_curve", [])) != insdel_steps + 1:
                raise ValueError(f"{image_result.get('image_name')} {method_name}: bad insertion curve")
            if len(metrics.get("deletion_curve", [])) != insdel_steps + 1:
                raise ValueError(f"{image_result.get('image_name')} {method_name}: bad deletion curve")
            if not math.isfinite(metrics.get("insertion_auc")):
                raise ValueError(f"{image_result.get('image_name')} {method_name}: bad insertion AUC")
            if not math.isfinite(metrics.get("deletion_auc")):
                raise ValueError(f"{image_result.get('image_name')} {method_name}: bad deletion AUC")


def csv_target(entry: dict[str, Any]) -> int:
    value = entry.get("resnet50_predicted_label_idx")
    if value is None:
        raise ValueError("--target-from-csv resnet50_pred requires predicted_label_idx")
    return int(value)


def selection_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "correct_index",
        "source_order",
        "ground_truth_label_idx",
        "ground_truth_label_name",
        "predicted_label_idx",
        "predicted_label_name",
        "top1_score",
        "selection_top1_score",
        "wnid",
    ):
        if key in entry:
            metadata[key] = entry.get(key)
    if "top1_score" not in metadata and entry.get("selection_top1_score") is not None:
        metadata["top1_score"] = entry.get("selection_top1_score")
    if "selection_top1_score" not in metadata and entry.get("top1_score") is not None:
        metadata["selection_top1_score"] = entry.get("top1_score")
    return metadata


def run_one_image(
    entry: dict[str, Any],
    backbone: torch.nn.Module,
    transform: T.Compose,
    categories: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = Path(entry["copied_image_path"] or entry["image_path"])
    x = load_image_tensor(image_path, transform, device)
    if args.target_class is not None:
        forced_target = args.target_class
    elif args.target_from_csv == "resnet50_pred":
        forced_target = csv_target(entry)
    else:
        forced_target = None
    target_class, target_score = select_target(backbone, x, forced_target)
    model = ClassLogitModel(backbone, target_class=target_class).to(device).eval()
    baseline = torch.zeros_like(x)
    methods = run_all_methods(
        model, x, baseline, N=args.steps, tau=args.tau,
        mu_iter=args.iters, lr=args.lr)

    target_name = categories[target_class] if categories else None
    image_result: dict[str, Any] = {
        "image_path": str(image_path),
        "image_name": entry["image_name"],
        "rank": entry.get("rank"),
        "resnet50_selection_score": entry.get("resnet50_selection_score"),
        "target_class": target_class,
        "target_score": target_score,
        "target_label_name": target_name,
        "success": True,
        "methods": {method.name: method_summary(method) for method in methods},
    }
    image_result.update(selection_metadata(entry))

    if args.insdel:
        insertion_deletion = run_insertion_deletion(
            model, x, baseline, methods, n_steps=args.insdel_steps,
            score_mode=args.auc_score, score_model=backbone,
            score_target_class=target_class)
        _validate_insertion_deletion_export(
            insertion_deletion, methods, args.insdel_steps)
        image_result["insertion_deletion"] = insertion_deletion

    validate_successful_image(image_result, args.insdel_steps, args.insdel)
    return image_result


def finite_values(values: list[Any]) -> list[float]:
    return [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


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
            for key in METHOD_SUMMARY_KEYS:
                values[method_name][key].append(metrics.get(key))
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
    aggregate["num_images_success"] = sum(1 for image in images if image.get("success"))
    aggregate["num_images_failed"] = sum(1 for image in images if not image.get("success"))
    return aggregate


def build_output(args: argparse.Namespace, entries: list[dict[str, Any]],
                 images: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "config": {
            "seed": args.seed,
            "model_name": args.model_name,
            "steps": args.steps,
            "tau": args.tau,
            "iters": args.iters,
            "lr": args.lr,
            "num_images": len(entries),
            "selection_csv": args.selected_csv,
            "image_dir": None if args.selected_csv else args.image_dir,
            "target_from_csv": args.target_from_csv,
            "target_class": args.target_class,
            "insdel": args.insdel,
            "insdel_steps": args.insdel_steps,
            "auc_score": args.auc_score,
            "skip_errors": args.skip_errors,
        },
        "images": images,
        "aggregate": compute_aggregate(images),
    }


def write_output(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tmp.replace(path)


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
    entries = (
        read_selected_csv(Path(args.selected_csv), args.num_images)
        if args.selected_csv
        else list_images(Path(args.image_dir), args.num_images)
    )
    output_path = Path(args.output_json) if args.output_json else default_output_path(args)

    backbone, transform, categories = load_backbone(args.model_name, device)
    images: list[dict[str, Any]] = []
    print(f"Model: {args.model_name}")
    print(f"Images: {len(entries)}")
    print(f"Output: {output_path}")

    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] rank={entry.get('rank')} {entry['image_name']}")
        try:
            images.append(run_one_image(entry, backbone, transform, categories, device, args))
        except Exception as exc:
            if not args.skip_errors:
                raise
            print(f"  ERROR: {exc}")
            images.append({
                "image_path": entry.get("copied_image_path") or entry.get("image_path"),
                "image_name": entry.get("image_name"),
                "rank": entry.get("rank"),
                "resnet50_selection_score": entry.get("resnet50_selection_score"),
                **selection_metadata(entry),
                "success": False,
                "error": str(exc),
            })
        write_output(output_path, build_output(args, entries, images))

    output = build_output(args, entries, images)
    write_output(output_path, output)
    print_summary_table(output["aggregate"])
    print(f"\nBatch results -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
