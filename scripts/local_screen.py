#!/usr/bin/env python3
"""Run one frozen submission across every public Easy and Medium suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ROOT = ROOT / ".benchmark" / "official"
DEFAULT_OUTPUT_ROOT = ROOT / ".benchmark" / "screens"
DATASETS = tuple(f"e{index}" for index in range(1, 6)) + tuple(
    f"m{index}" for index in range(1, 6)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "submission.py")
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="mps:0")
    parser.add_argument("--easy-seconds", type=float, default=30.0)
    parser.add_argument("--medium-seconds", type=float, default=60.0)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    return parser.parse_args()


def local_manifest(
    source: Path,
    *,
    device: str,
    training_seconds: float,
) -> dict:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["name"] = f"{manifest['name']}-local-screen"
    manifest["data"]["num_workers"] = 0
    manifest["data"]["pin_memory"] = False
    manifest["runtime"]["device"] = device
    manifest["runtime"]["dtype"] = "float32"
    manifest["runtime"]["amp"] = False
    manifest["runtime"]["compile"] = False
    manifest["runtime"]["total_training_time_seconds"] = training_seconds
    manifest["runtime"]["log_every"] = 100
    return manifest


def run_dataset(args: argparse.Namespace, dataset: str) -> dict:
    tier = "easy" if dataset.startswith("e") else "medium"
    seconds = args.easy_seconds if tier == "easy" else args.medium_seconds
    source = (
        args.official_root / "benchmark" / "manifests" / f"h100_{tier}_{dataset}.json"
    )
    output_dir = args.output_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{dataset}_manifest.json"
    log_path = output_dir / f"{dataset}.log"
    manifest_path.write_text(
        json.dumps(
            local_manifest(
                source,
                device=args.device,
                training_seconds=seconds,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "benchmark.runner",
        "--manifest",
        str(manifest_path),
        "--submission-file",
        str(args.submission.resolve()),
        "--include-structured-metrics",
    ]
    print(f"screening {dataset} for {seconds:.0f}s", flush=True)
    process = subprocess.run(
        command,
        cwd=args.official_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = process.stdout + process.stderr
    log_path.write_text(output, encoding="utf-8")
    result_line = next(
        (
            line
            for line in reversed(output.splitlines())
            if line.startswith("RESULT_JSON=")
        ),
        None,
    )
    if process.returncode or result_line is None:
        tail = "\n".join(output.splitlines()[-12:])
        print(
            f"{dataset} failed with exit code {process.returncode}\n{tail}", flush=True
        )
        return {
            "dataset": dataset,
            "status": "failed",
            "exit_code": process.returncode,
            "log": str(log_path),
        }
    result = json.loads(result_line.removeprefix("RESULT_JSON="))
    score = result["score"]["mean_exact_accuracy"]
    seed = result["seeds"][0]
    evaluation = {
        split: metrics["exact_accuracy"]
        for split, metrics in seed["evaluation"].items()
    }
    summary = {
        "dataset": dataset,
        "status": "succeeded",
        "score": score,
        "completed_training_steps": seed["completed_training_steps"],
        "evaluation": evaluation,
        "max_certified_time_steps": seed["depth_profile"]["max_certified_time_steps"],
        "ood_n_max_certified_time_steps": seed["depth_profile"][
            "ood_n_max_certified_time_steps"
        ],
        "log": str(log_path),
    }
    split_scores = " ".join(
        f"{split}={accuracy:.2%}" for split, accuracy in evaluation.items()
    )
    print(
        f"{dataset} score={score:.2%} steps={seed['completed_training_steps']} "
        f"{split_scores}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    datasets = tuple(args.dataset) if args.dataset else DATASETS
    results = [run_dataset(args, dataset) for dataset in datasets]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    failures = [result for result in results if result["status"] != "succeeded"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
