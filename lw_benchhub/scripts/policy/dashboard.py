"""Streamlit dashboard for LW-BenchHub benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _find_runs(results_root: Path) -> list[Path]:
    if (results_root / "summary.json").exists():
        return [results_root]
    runs = []
    for path in sorted(results_root.glob("**/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        runs.append(path.parent)
    return runs


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten(prefix: str, value: Any, row: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, row)
    elif isinstance(value, list):
        row[prefix] = json.dumps(value) if any(isinstance(item, (dict, list)) for item in value) else value
    else:
        row[prefix] = value


def _rollouts_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rollout in summary.get("rollouts", []):
        row: dict[str, Any] = {}
        _flatten("", rollout, row)
        rows.append(row)
    return pd.DataFrame(rows)


def _path_from_value(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _display_images(title: str, images: dict[str, str]) -> None:
    st.subheader(title)
    if not images:
        st.caption("No images exported for this rollout.")
        return
    cols = st.columns(min(len(images), 3))
    for idx, (camera, image_path) in enumerate(images.items()):
        with cols[idx % len(cols)]:
            path = _path_from_value(image_path)
            if path:
                st.image(str(path), caption=camera, use_container_width=True)
            else:
                st.caption(f"{camera}: missing")


def main(results_root: Path) -> None:
    st.set_page_config(page_title="LW-BenchHub Results", layout="wide")
    st.title("LW-BenchHub Results")

    runs = _find_runs(results_root)
    if not runs:
        st.error(f"No summary.json files found under {results_root}")
        return

    run_labels = [str(path) for path in runs]
    selected_run = Path(st.sidebar.selectbox("Run", run_labels))
    summary = _load_json(selected_run / "summary.json")
    df = _rollouts_frame(summary)

    st.sidebar.caption(f"Loaded {selected_run}")
    task_options = ["All"] + sorted(v for v in df.get("task", pd.Series(dtype=str)).dropna().unique())
    robot_options = ["All"] + sorted(v for v in df.get("robot", pd.Series(dtype=str)).dropna().unique())
    task_filter = st.sidebar.selectbox("Task", task_options)
    robot_filter = st.sidebar.selectbox("Robot", robot_options)

    filtered = df.copy()
    if task_filter != "All" and "task" in filtered:
        filtered = filtered[filtered["task"] == task_filter]
    if robot_filter != "All" and "robot" in filtered:
        filtered = filtered[filtered["robot"] == robot_filter]

    total = len(filtered)
    successes = int(filtered["success"].fillna(False).sum()) if "success" in filtered else 0
    failures = total - successes
    success_rate = successes / total if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes", total)
    c2.metric("Success Rate", f"{success_rate:.1%}")
    c3.metric("Successes", successes)
    c4.metric("Failures", failures)

    st.subheader("Metrics")
    if filtered.empty:
        st.info("No rollout rows match the current filters.")
        return

    preferred = ["episode_index", "success", "time_steps", "task", "robot", "layout", "video_path"]
    ordered_cols = [col for col in preferred if col in filtered.columns]
    ordered_cols.extend(col for col in filtered.columns if col not in ordered_cols)
    st.dataframe(filtered[ordered_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered CSV",
        filtered[ordered_cols].to_csv(index=False),
        file_name="lw_benchhub_filtered_metrics.csv",
        mime="text/csv",
    )

    st.subheader("Rollout")
    episode_labels = [
        f"{int(row.episode_index):03d} | {'success' if bool(row.success) else 'failure'} | {getattr(row, 'task', '')}"
        for row in filtered.itertuples(index=False)
    ]
    selected_label = st.selectbox("Episode", episode_labels)
    selected_pos = episode_labels.index(selected_label)
    rollout = filtered.iloc[selected_pos].to_dict()
    episode_index = int(rollout.get("episode_index", selected_pos))
    raw_rollouts = summary.get("rollouts", [])
    raw_rollout = next(
        (item for item in raw_rollouts if int(item.get("episode_index", -1)) == episode_index),
        raw_rollouts[selected_pos] if selected_pos < len(raw_rollouts) else {},
    )

    video_path = _path_from_value(raw_rollout.get("video_path"))
    if video_path:
        st.video(str(video_path))
    else:
        st.caption("No rollout video exported for this episode.")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        _display_images("Init Images", raw_rollout.get("init_images", {}))
    with img_col2:
        _display_images("End Images", raw_rollout.get("end_images", {}))

    with st.expander("Raw Rollout Metadata"):
        st.json(raw_rollout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View LW-BenchHub benchmark results.")
    parser.add_argument("--results_dir", default="./results", type=Path)
    args = parser.parse_args()
    main(args.results_dir)
