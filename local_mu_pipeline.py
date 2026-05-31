#!/usr/bin/env python3
"""Local/CLI pipeline for Mu-Optimized ImageNet evaluation.

This script reproduces the workflow from the Kaggle notebook, but uses local relative paths by default:
1. Prepare the first-N ImageNet validation images.
2. Filter images where ResNet50 prediction equals the ground-truth label.
3. Run IG, IDG-PDF, and μ-Optimized on ResNet50 and VGG16.
4. Flatten insertion/deletion JSON results to CSV.
5. Select correct_index lists where μ-Optimized beats IG and IDG-PDF on both models.
6. Optionally archive results/logs for local backup.

Run from the repository root after cloning Tamnt240904/Mu-Optimized.
Example:
    python local_mu_pipeline.py --first-count 200 --num-eval-images 10 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable


DEFAULT_MODELS = ["resnet50", "vgg16"]


def default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mu-Optimized local pipeline from CLI.")

    parser.add_argument("--repo-dir", type=Path, default=Path.cwd(), help="Repository root. Default: current directory.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Local data directory. Default: ./data")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Local results directory. Default: ./results")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="Local logs directory. Default: ./logs")

    parser.add_argument("--first-count", type=int, default=5000, help="Number of first ImageNet validation candidates to prepare.")
    parser.add_argument("--prep-batch-size", type=int, default=64)
    parser.add_argument("--device", default=default_device())

    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=["resnet50", "vgg16", "densenet121"])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--insdel-steps", type=int, default=50)
    parser.add_argument("--auc-score", choices=["logit", "confidence"], default="confidence")

    parser.add_argument(
        "--num-eval-images",
        type=int,
        default=None,
        help="Optional limit for full evaluation. If omitted, batch_eval.py evaluates all selected rows.",
    )

    parser.add_argument("--run-small-test", action="store_true", help="Run a small smoke test before full evaluation.")
    parser.add_argument("--test-first-count", type=int, default=50)
    parser.add_argument("--test-num-images", type=int, default=3)
    parser.add_argument("--test-steps", type=int, default=16)
    parser.add_argument("--test-iters", type=int, default=20)
    parser.add_argument("--test-insdel-steps", type=int, default=10)

    parser.add_argument("--skip-prepare", action="store_true", help="Use an existing selected.csv.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip full batch_eval and use existing JSON files.")
    parser.add_argument("--skip-flatten", action="store_true", help="Skip flattening and use existing flat CSV files.")
    parser.add_argument("--skip-selection", action="store_true", help="Skip selection step.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip commands whose output file already exists and is non-empty.")
    parser.add_argument("--resume-prepare", action="store_true", help="Pass --resume to prepare_firstN_correct.py.")
    parser.add_argument("--archive", action="store_true", help="Create a zip archive of results and logs at the end.")
    parser.add_argument("--archive-dir", type=Path, default=Path("archives"), help="Directory for local zip archives. Default: ./archives")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")

    return parser.parse_args()


def ensure_repo_files(repo_dir: Path) -> None:
    required = [
        "batch_eval.py",
        "prepare_firstN_correct.py",
        "flatten_insdel_json.py",
        "select_mu_better_two_models.py",
    ]
    missing = [name for name in required if not (repo_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required repo files: " + ", ".join(missing) + f". repo_dir={repo_dir}"
        )


def run_streaming(cmd: list[str], log_path: Path | None = None, cwd: Path | None = None) -> None:
    """Run a command, stream stdout/stderr to console, and optionally write the same output to a log."""
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print("\n" + "=" * 100)
    print(printable)
    print("=" * 100)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
    else:
        log_file = None

    start = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
        ret = process.wait()
    finally:
        if log_file is not None:
            log_file.close()

    elapsed = time.time() - start
    print(f"\nFinished ret={ret}, elapsed={elapsed / 60:.2f} min")
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {printable}")


def output_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def maybe_run(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    skip_existing: bool = False,
    expected_output: Path | None = None,
    tolerate_failure_if_output_exists: bool = False,
    dry_run: bool = False,
) -> None:
    if skip_existing and expected_output is not None and output_exists(expected_output):
        print(f"SKIP existing output: {expected_output}")
        return
    if dry_run:
        printable = " ".join(shlex.quote(str(x)) for x in cmd)
        print("\nDRY RUN:", printable)
        if expected_output is not None:
            print("Expected output:", expected_output)
        return
    try:
        run_streaming(cmd, log_path=log_path, cwd=cwd)
    except RuntimeError:
        if tolerate_failure_if_output_exists and expected_output is not None and output_exists(expected_output):
            print(
                "WARNING: command failed after producing expected output; "
                f"continuing with {expected_output}"
            )
            return
        raise


def count_selected_rows(selected_csv: Path) -> int:
    if not selected_csv.exists():
        raise FileNotFoundError(f"selected.csv not found: {selected_csv}")
    with selected_csv.open("r", encoding="utf-8") as f:
        # subtract header
        return max(sum(1 for _ in f) - 1, 0)


def zip_paths(archive_path: Path, paths: Iterable[Path]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if not path.exists():
                continue
            if path.is_file():
                zf.write(path, arcname=str(path))
            else:
                for file in path.rglob("*"):
                    if file.is_file():
                        zf.write(file, arcname=str(file))
    print(f"Archive: {archive_path}")
    print(f"Archive size: {archive_path.stat().st_size / (1024 ** 2):.2f} MB")


def main() -> None:
    args = parse_args()

    repo_dir = args.repo_dir.resolve()
    os.chdir(repo_dir)
    ensure_repo_files(repo_dir)

    data_root = args.data_root
    results_dir = args.results_dir
    logs_dir = args.logs_dir

    for d in [data_root, results_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    full_root = data_root / f"imagenet_first{args.first_count}_correct"
    test_root = data_root / f"imagenet_first{args.test_first_count}_correct_test"

    print("Repo:", repo_dir)
    print("Data root:", data_root)
    print("Results:", results_dir)
    print("Logs:", logs_dir)
    print("Full root:", full_root)

    # 1. Optional smoke test.
    if args.run_small_test:
        prepare_test_cmd = [
            sys.executable,
            "-u",
            "prepare_firstN_correct.py",
            "--first-count",
            str(args.test_first_count),
            "--output-root",
            str(test_root),
            "--batch-size",
            "16",
            "--device",
            args.device,
        ]
        if args.resume_prepare:
            prepare_test_cmd.append("--resume")
        maybe_run(
            prepare_test_cmd,
            logs_dir / f"test_prepare_first{args.test_first_count}.log",
            cwd=repo_dir,
            skip_existing=args.skip_existing,
            expected_output=test_root / "selected.csv",
            tolerate_failure_if_output_exists=True,
            dry_run=args.dry_run,
        )
        if not args.dry_run or (test_root / "selected.csv").exists():
            print("Small test selected rows:", count_selected_rows(test_root / "selected.csv"))

        for model in args.models:
            out = results_dir / "test" / f"test_{model}_first{args.test_first_count}_N{args.test_steps}_tau{args.tau}_{args.auc_score}.json"
            log = logs_dir / f"test_{model}_first{args.test_first_count}_N{args.test_steps}_tau{args.tau}_{args.auc_score}.log"
            cmd = [
                sys.executable,
                "-u",
                "batch_eval.py",
                "--selected-csv",
                str(test_root / "selected.csv"),
                "--num-images",
                str(args.test_num_images),
                "--model-name",
                model,
                "--steps",
                str(args.test_steps),
                "--tau",
                str(args.tau),
                "--iters",
                str(args.test_iters),
                "--lr",
                str(args.lr),
                "--insdel",
                "--insdel-steps",
                str(args.test_insdel_steps),
                "--auc-score",
                args.auc_score,
                "--device",
                args.device,
                "--skip-errors",
                "--output-json",
                str(out),
            ]
            maybe_run(
                cmd,
                log,
                cwd=repo_dir,
                skip_existing=args.skip_existing,
                expected_output=out,
                dry_run=args.dry_run,
            )

    # 2. Full prepare.
    selected_csv = full_root / "selected.csv"
    if not args.skip_prepare:
        prepare_cmd = [
            sys.executable,
            "-u",
            "prepare_firstN_correct.py",
            "--first-count",
            str(args.first_count),
            "--output-root",
            str(full_root),
            "--batch-size",
            str(args.prep_batch_size),
            "--device",
            args.device,
        ]
        if args.resume_prepare:
            prepare_cmd.append("--resume")
        maybe_run(
            prepare_cmd,
            logs_dir / f"full_prepare_first{args.first_count}.log",
            cwd=repo_dir,
            skip_existing=args.skip_existing,
            expected_output=selected_csv,
            tolerate_failure_if_output_exists=True,
            dry_run=args.dry_run,
        )
    else:
        print("skip_prepare=True; using existing", selected_csv)

    if args.dry_run and not selected_csv.exists():
        print("ResNet50-correct selected rows: unknown (dry run; selected.csv not present)")
    else:
        n_correct = count_selected_rows(selected_csv)
        print("ResNet50-correct selected rows:", n_correct)

    # 3. Full eval.
    full_json_paths: dict[str, Path] = {}
    if not args.skip_eval:
        (results_dir / "full").mkdir(parents=True, exist_ok=True)
        for model in args.models:
            out = results_dir / "full" / f"full_{model}_first{args.first_count}_correct_N{args.steps}_tau{args.tau}_{args.auc_score}.json"
            log = logs_dir / f"full_{model}_first{args.first_count}_correct_N{args.steps}_tau{args.tau}_{args.auc_score}.log"
            cmd = [
                sys.executable,
                "-u",
                "batch_eval.py",
                "--selected-csv",
                str(selected_csv),
                "--model-name",
                model,
                "--steps",
                str(args.steps),
                "--tau",
                str(args.tau),
                "--iters",
                str(args.iters),
                "--lr",
                str(args.lr),
                "--insdel",
                "--insdel-steps",
                str(args.insdel_steps),
                "--auc-score",
                args.auc_score,
                "--device",
                args.device,
                "--skip-errors",
                "--output-json",
                str(out),
            ]
            if args.num_eval_images is not None:
                cmd[cmd.index("--model-name"):cmd.index("--model-name")] = ["--num-images", str(args.num_eval_images)]
            maybe_run(
                cmd,
                log,
                cwd=repo_dir,
                skip_existing=args.skip_existing,
                expected_output=out,
                dry_run=args.dry_run,
            )
            full_json_paths[model] = out
    else:
        print("skip_eval=True; using existing full JSON files.")
        for model in args.models:
            full_json_paths[model] = results_dir / "full" / f"full_{model}_first{args.first_count}_correct_N{args.steps}_tau{args.tau}_{args.auc_score}.json"

    # 4. Flatten.
    flat_paths: dict[str, Path] = {}
    if not args.skip_flatten:
        (results_dir / "full").mkdir(parents=True, exist_ok=True)
        for model in args.models:
            in_json = full_json_paths[model]
            out_csv = results_dir / "full" / f"full_{model}_first{args.first_count}_correct_N{args.steps}_tau{args.tau}_{args.auc_score}_flat.csv"
            cmd = [
                sys.executable,
                "-u",
                "flatten_insdel_json.py",
                "--input-json",
                str(in_json),
                "--output-csv",
                str(out_csv),
            ]
            maybe_run(
                cmd,
                logs_dir / f"flatten_{model}.log",
                cwd=repo_dir,
                skip_existing=args.skip_existing,
                expected_output=out_csv,
                dry_run=args.dry_run,
            )
            flat_paths[model] = out_csv
    else:
        print("skip_flatten=True; using existing flat CSV files.")
        for model in args.models:
            flat_paths[model] = results_dir / "full" / f"full_{model}_first{args.first_count}_correct_N{args.steps}_tau{args.tau}_{args.auc_score}_flat.csv"

    # 5. Selection requires resnet50 and vgg16.
    if not args.skip_selection:
        if "resnet50" not in flat_paths or "vgg16" not in flat_paths:
            print("Skipping selection: requires both resnet50 and vgg16 flat CSVs.")
        else:
            selection_dir = results_dir / "selection" / f"first{args.first_count}_N{args.steps}_tau{args.tau}_{args.auc_score}"
            cmd = [
                sys.executable,
                "-u",
                "select_mu_better_two_models.py",
                "--resnet-csv",
                str(flat_paths["resnet50"]),
                "--vgg-csv",
                str(flat_paths["vgg16"]),
                "--output-dir",
                str(selection_dir),
            ]
            maybe_run(
                cmd,
                logs_dir / "select_mu_better_two_models.log",
                cwd=repo_dir,
                skip_existing=args.skip_existing,
                expected_output=selection_dir / "merged_per_image_metrics.csv",
                dry_run=args.dry_run,
            )
            print("Selection dir:", selection_dir)
    else:
        print("skip_selection=True; skipping selection.")

    # 6. Archive.
    if args.archive:
        args.archive_dir.mkdir(parents=True, exist_ok=True)
        archive = args.archive_dir / f"mu_pipeline_results_first{args.first_count}_N{args.steps}_tau{args.tau}_{args.auc_score}.zip"
        zip_paths(archive, [results_dir, logs_dir, selected_csv])

    print("Pipeline finished.")
    print("Local outputs:")
    print("  data:", data_root)
    print("  results:", results_dir)
    print("  logs:", logs_dir)


if __name__ == "__main__":
    main()
