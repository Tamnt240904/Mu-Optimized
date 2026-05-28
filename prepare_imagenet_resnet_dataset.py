#!/usr/bin/env python3
"""Prepare ResNet50-selected ImageNet subsets for repeated experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import traceback
import urllib.request
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/"
    "image-models/imagenet_class_index.json"
)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each seed, sample ImageNet validation candidates, score them "
            "with torchvision ResNet50, and save the top correctly classified "
            "images plus selected.csv."
        )
    )
    parser.add_argument(
        "--dataset-id",
        default="Tsomaros/Imagenet-1k_validation",
        help=(
            "Hugging Face dataset id used when --imagenet-root is omitted. "
            "Use ILSVRC/imagenet-1k if your HF token has gated access."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--select-count", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=None,
                        help="Backward-compatible alias for --candidate-count.")
    parser.add_argument("--keep-count", type=int, default=None,
                        help="Backward-compatible alias for --select-count.")
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/imagenet_resnet50_selected"))
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=None,
        help=(
            "Optional local ImageNet validation root. Supports ImageFolder "
            "layout with WNID class folders or flat files whose names include "
            "an ImageNet WNID."
        ),
    )
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    parser.add_argument("--class-index-json", type=Path,
                        default=Path("data/imagenet_class_index.json"))
    parser.add_argument("--resume", action="store_true")

    # Backward-compatible single-output options from the earlier helper.
    parser.add_argument("--random-dir", type=Path, default=None)
    parser.add_argument("--selected-dir", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imagenet_class_index(path: Path) -> tuple[dict[str, int], dict[int, tuple[str, str]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(CLASS_INDEX_URL, timeout=60) as response:
            path.write_bytes(response.read())
    raw = json.loads(path.read_text(encoding="utf-8"))
    idx_to_meta = {
        int(index): (value[0], value[1].replace("_", " "))
        for index, value in raw.items()
    }
    wnid_to_idx = {wnid: index for index, (wnid, _) in idx_to_meta.items()}
    return wnid_to_idx, idx_to_meta


def label_to_wnid(dataset, label: int | str) -> str:
    feature = dataset.features.get("label") if dataset.features else None
    if feature is not None and hasattr(feature, "int2str"):
        return feature.int2str(int(label))
    if isinstance(label, str):
        return label
    raise ValueError("Dataset labels must be WNIDs or ClassLabel WNID names.")


def wnid_from_path(path: Path, wnid_to_idx: dict[str, int]) -> str | None:
    if path.parent.name in wnid_to_idx:
        return path.parent.name
    for part in path.stem.replace("-", "_").split("_"):
        if part in wnid_to_idx:
            return part
    return None


def save_rgb_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=95)


def sample_from_hf(
    dataset_id: str,
    split: str,
    seed: int,
    candidate_count: int,
    shuffle_buffer: int,
    candidate_dir: Path,
    wnid_to_idx: dict[str, int],
    resume: bool,
) -> list[dict[str, object]]:
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    shuffled = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    records: list[dict[str, object]] = []

    progress = tqdm(total=candidate_count, desc=f"Seed {seed}: download candidates")
    for source_order, row in enumerate(shuffled):
        if len(records) >= candidate_count:
            break
        wnid = label_to_wnid(dataset, row["label"])
        if wnid not in wnid_to_idx:
            raise KeyError(f"WNID {wnid!r} is not in ImageNet class index")
        label_idx = wnid_to_idx[wnid]
        name = f"seed{seed}_candidate_{len(records):05d}_{wnid}_c{label_idx:04d}.JPEG"
        image_path = candidate_dir / name
        if not resume or not image_path.exists():
            save_rgb_jpeg(row["image"], image_path)
        records.append({
            "seed": seed,
            "image_path": image_path,
            "image_name": name,
            "source_order": source_order,
            "ground_truth_label_idx": label_idx,
            "wnid": wnid,
        })
        progress.update(1)

    progress.close()
    if len(records) < candidate_count:
        raise RuntimeError(
            f"Dataset ended after {len(records)} candidates for seed {seed}; "
            f"expected {candidate_count}."
        )
    return records


def iter_local_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def sample_from_local(
    imagenet_root: Path,
    seed: int,
    candidate_count: int,
    wnid_to_idx: dict[str, int],
) -> list[dict[str, object]]:
    all_paths = sorted(iter_local_images(imagenet_root))
    rng = random.Random(seed)
    rng.shuffle(all_paths)

    records: list[dict[str, object]] = []
    for path in all_paths:
        wnid = wnid_from_path(path, wnid_to_idx)
        if wnid is None:
            continue
        records.append({
            "seed": seed,
            "image_path": path,
            "image_name": path.name,
            "source_order": len(records),
            "ground_truth_label_idx": wnid_to_idx[wnid],
            "wnid": wnid,
        })
        if len(records) >= candidate_count:
            break

    if len(records) < candidate_count:
        raise RuntimeError(
            f"Only found {len(records)} labeled local images in {imagenet_root}; "
            f"expected {candidate_count}."
        )
    return records


def load_resnet50(device: torch.device):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms(), weights.meta["categories"]


@torch.no_grad()
def score_correct_resnet50(
    records: list[dict[str, object]],
    batch_size: int,
    device: torch.device,
    idx_to_meta: dict[int, tuple[str, str]],
) -> list[dict[str, object]]:
    model, transform, categories = load_resnet50(device)
    correct: list[dict[str, object]] = []

    for start in tqdm(range(0, len(records), batch_size), desc="Score ResNet50"):
        batch = records[start:start + batch_size]
        tensors = []
        for record in batch:
            with Image.open(record["image_path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        x = torch.stack(tensors).to(device)
        probs = F.softmax(model(x), dim=1)
        scores, preds = probs.max(dim=1)

        for record, pred, score in zip(batch, preds.tolist(), scores.tolist()):
            gt_idx = int(record["ground_truth_label_idx"])
            if int(pred) != gt_idx:
                continue
            enriched = dict(record)
            enriched["predicted_label_idx"] = int(pred)
            enriched["predicted_label_name"] = categories[int(pred)]
            enriched["ground_truth_label_name"] = idx_to_meta[gt_idx][1]
            enriched["top1_score"] = float(score)
            correct.append(enriched)

    correct.sort(key=lambda item: float(item["top1_score"]), reverse=True)
    return correct


def seed_paths(args: argparse.Namespace, seed: int) -> tuple[Path, Path, Path]:
    if args.selected_dir is not None or args.csv_path is not None or args.random_dir is not None:
        candidate_dir = args.random_dir or Path("generated_imagenet/imagenet_random_5000")
        selected_dir = args.selected_dir or Path("generated_imagenet/imagenet_resnet50_correct")
        csv_path = args.csv_path or selected_dir.parent / "selected.csv"
        return candidate_dir, selected_dir, csv_path

    seed_root = args.output_root / f"seed_{seed}"
    return seed_root / "candidates", seed_root / "images", seed_root / "selected.csv"


def export_selected(
    seed: int,
    correct: list[dict[str, object]],
    select_count: int,
    selected_dir: Path,
    csv_path: Path,
) -> None:
    if len(correct) < select_count:
        raise RuntimeError(
            f"Seed {seed}: only {len(correct)} correct ResNet50 predictions; "
            f"cannot select {select_count}."
        )

    selected_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, record in enumerate(correct[:select_count], start=1):
        source = Path(record["image_path"])
        copied = selected_dir / source.name
        if source.resolve() != copied.resolve():
            shutil.copy2(source, copied)
        rows.append({
            "seed": seed,
            "rank": rank,
            "image_path": str(source),
            "copied_image_path": str(copied),
            "image_name": copied.name,
            "ground_truth_label_idx": int(record["ground_truth_label_idx"]),
            "ground_truth_label_name": record["ground_truth_label_name"],
            "predicted_label_idx": int(record["predicted_label_idx"]),
            "predicted_label_name": record["predicted_label_name"],
            "top1_score": f"{float(record['top1_score']):.10f}",
            "wnid": record["wnid"],
        })

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(args: argparse.Namespace, seed: int, device: torch.device,
             wnid_to_idx: dict[str, int],
             idx_to_meta: dict[int, tuple[str, str]]) -> None:
    candidate_dir, selected_dir, csv_path = seed_paths(args, seed)
    print(f"\nSeed {seed}")
    print(f"Candidates -> {candidate_dir}")
    print(f"Selected   -> {selected_dir}")
    print(f"CSV        -> {csv_path}")

    if args.imagenet_root is None:
        records = sample_from_hf(
            args.dataset_id, args.split, seed, args.candidate_count,
            args.shuffle_buffer, candidate_dir, wnid_to_idx, args.resume)
    else:
        records = sample_from_local(
            args.imagenet_root, seed, args.candidate_count, wnid_to_idx)

    correct = score_correct_resnet50(records, args.batch_size, device, idx_to_meta)
    export_selected(seed, correct, args.select_count, selected_dir, csv_path)
    print(f"Seed {seed}: candidates={len(records)}, correct={len(correct)}, "
          f"selected={args.select_count}")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.candidate_count = args.candidate_count or args.sample_count or 5000
    args.select_count = args.select_count or args.keep_count or 200
    if args.select_count > args.candidate_count:
        raise ValueError("--select-count must be <= --candidate-count")
    if args.seeds is None:
        args.seeds = [args.seed]
    return args


def main() -> None:
    args = normalize_args(parse_args())
    device = choose_device(args.device)
    wnid_to_idx, idx_to_meta = load_imagenet_class_index(args.class_index_json)
    print(f"Device: {device}")
    for seed in args.seeds:
        run_seed(args, seed, device, wnid_to_idx, idx_to_meta)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
