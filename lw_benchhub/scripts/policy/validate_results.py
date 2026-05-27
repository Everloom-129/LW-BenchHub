"""Validate LW-BenchHub benchmark result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported LW-BenchHub benchmark results.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--require-example", choices=["success", "failure"], action="append", default=[])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(results_dir: Path, required_examples: list[str]) -> list[str]:
    errors: list[str] = []
    summary_path = results_dir / "summary.json"
    metrics_path = results_dir / "metrics.csv"

    require(summary_path.exists(), f"missing {summary_path}", errors)
    require(metrics_path.exists(), f"missing {metrics_path}", errors)
    if errors:
        return errors

    summary = load_json(summary_path)
    rollouts = summary.get("rollouts", [])
    require(isinstance(rollouts, list) and bool(rollouts), "summary.json has no rollouts", errors)
    require(summary.get("test_count") == len(rollouts), "test_count does not match rollouts length", errors)

    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        metrics_rows = list(csv.DictReader(f))
    require(len(metrics_rows) == len(rollouts), "metrics.csv row count does not match rollouts length", errors)

    for rollout in rollouts:
        idx = rollout.get("episode_index", "?")
        video_path = Path(str(rollout.get("video_path", "")))
        require(video_path.exists(), f"rollout {idx} missing video {video_path}", errors)
        require(video_path.suffix.lower() == ".mp4", f"rollout {idx} video is not mp4: {video_path}", errors)
        require(isinstance(rollout.get("success"), bool), f"rollout {idx} success is not boolean", errors)
        require(isinstance(rollout.get("time_steps"), int), f"rollout {idx} time_steps is not int", errors)
        init_images = rollout.get("init_images", {})
        end_images = rollout.get("end_images", {})
        require(bool(init_images), f"rollout {idx} has no init images", errors)
        require(bool(end_images), f"rollout {idx} has no end images", errors)
        for label, images in (("init", init_images), ("end", end_images)):
            for camera, image_path in images.items():
                path = Path(str(image_path))
                require(path.exists(), f"rollout {idx} missing {label} image {camera}: {path}", errors)

    examples = summary.get("examples", {})
    for example in required_examples:
        path = Path(str(examples.get(example, "")))
        require(path.exists(), f"missing required {example} example directory", errors)
        require((path / "record_video.mp4").exists(), f"missing required {example} example video", errors)
        require((path / "metadata.json").exists(), f"missing required {example} example metadata", errors)

    return errors


def main() -> None:
    args = parse_args()
    errors = validate(args.results_dir, args.require_example)
    if errors:
        print("Result validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Result validation passed: {args.results_dir}")


if __name__ == "__main__":
    main()
