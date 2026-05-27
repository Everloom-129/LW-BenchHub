"""Run simple remote-env benchmark rollouts and export dashboard artifacts.

This script does not require a policy checkpoint. It connects to the running
LW-BenchHub env server, attaches the requested task/robot/scene, executes zero
or random actions, and exports the same results layout consumed by
``dashboard.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.append(str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LW-BenchHub remote-env benchmark rollouts.")
    parser.add_argument("--output_dir", default="./results/droid_franka_eval")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--action_mode", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record_camera", nargs="+", default=["global_camera_rgb", "hand_camera_rgb"])
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--task", default="LiftObj")
    parser.add_argument("--layout", default="robocasakitchen-9-8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--ipc_host", default="127.0.0.1")
    parser.add_argument("--ipc_port", type=int, default=50000)
    parser.add_argument("--ipc_authkey", default="lightwheel")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not pairs:
        return result
    if len(pairs) % 2:
        raise ValueError("--overrides must contain KEY VALUE pairs")
    for i in range(0, len(pairs), 2):
        key = pairs[i].lstrip("-")
        value: Any = pairs[i + 1]
        try:
            import yaml

            value = yaml.safe_load(value)
        except Exception:
            pass
        target = result
        parts = key.split(":")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def default_env_cfg(seed: int) -> dict[str, Any]:
    return {
        "task": "LiftObj",
        "robot": "Panda",
        "layout": "robocasakitchen-9-8",
        "rl": None,
        "scene_backend": "robocasa",
        "task_backend": "robocasa",
        "device": "cuda:0",
        "num_envs": 1,
        "usd_simplify": False,
        "enable_cameras": True,
        "video": True,
        "disable_fabric": False,
        "robot_scale": 1.0,
        "first_person_view": False,
        "seed": seed,
        "sources": None,
        "object_projects": None,
        "for_rl": False,
        "variant": "Visual",
        "concatenate_terms": False,
        "distributed": False,
        "execute_mode": "eval",
        "replay_cfgs": {"add_camera_to_observation": True},
    }


def deep_merge(original: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(original.get(key), dict) and isinstance(value, dict):
            deep_merge(original[key], value)
        else:
            original[key] = value
    return original


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return to_jsonable(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return repr(value)


def camera_image_from_obs(obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
    matching_keys = [key for key in policy_obs.keys() if key.startswith(camera_name)]
    if not matching_keys:
        return None
    image = policy_obs[matching_keys[0]]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "cpu"):
        image = image.cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def image_to_uint8(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "cpu"):
        image = image.cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def render_scene(env) -> np.ndarray | None:
    try:
        return image_to_uint8(env.render())
    except Exception:
        return None


def write_obs_images(obs: dict[str, Any], camera_names: list[str], image_dir: Path, env=None) -> dict[str, str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for camera_name in camera_names:
        image = camera_image_from_obs(obs, camera_name)
        if image is None:
            continue
        import mediapy as media

        image_path = image_dir / f"{camera_name}.png"
        media.write_image(image_path, image)
        paths[camera_name] = str(image_path)
    if env is not None:
        scene_image = render_scene(env)
        if scene_image is not None:
            import mediapy as media

            image_path = image_dir / "scene.png"
            media.write_image(image_path, scene_image)
            paths["scene"] = str(image_path)
    return paths


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return image[y_idx][:, x_idx]


def combine_camera_frame(obs: dict[str, Any], camera_names: list[str], height: int, width: int, env=None) -> np.ndarray | None:
    frames = []
    for camera_name in camera_names:
        image = camera_image_from_obs(obs, camera_name)
        if image is not None:
            frames.append(image)
    if not frames and env is not None:
        scene_image = render_scene(env)
        if scene_image is not None:
            frames.append(scene_image)
    if not frames:
        return np.zeros((height, width * max(1, len(camera_names)), 3), dtype=np.uint8)
    resized = [resize_nearest(frame, height, width) for frame in frames]
    combined = np.concatenate(resized, axis=1)
    target_width = width * max(1, len(camera_names))
    if combined.shape[1] != target_width:
        combined = resize_nearest(combined, height, target_width)
    return combined


def checker_results(env) -> dict[str, Any]:
    try:
        return to_jsonable(env._svc.call("unwrapped.cfg.isaaclab_arena_env.task.get_checker_results"))
    except Exception as exc:
        return {"_error": f"checker_results_unavailable: {exc}"}


def success_from_step(terminated: Any, extras: dict[str, Any]) -> bool:
    extras_json = to_jsonable(extras)
    if isinstance(extras_json, dict):
        for key in ("is_success", "success"):
            if key in extras_json:
                value = extras_json[key]
                if isinstance(value, list):
                    return any(bool(item) for item in value)
                return bool(value)
    if hasattr(terminated, "any"):
        import torch

        return bool(torch.as_tensor(terminated).any().item())
    return bool(terminated)


def action_for_space(env, mode: str):
    import torch

    shape = env.action_space.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "random":
        try:
            sample = env.action_space.sample()
            if not torch.is_tensor(sample):
                sample = torch.as_tensor(sample, dtype=torch.float32)
            return sample
        except Exception:
            return torch.rand(shape, device=device) * 2.0 - 1.0
    return torch.zeros(shape, device=device)


def flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, output)
    elif isinstance(value, list):
        output[prefix] = value if len(value) <= 8 else json.dumps(value)
    else:
        output[prefix] = value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat: dict[str, Any] = {}
        flatten("", row, flat)
        flat_rows.append(flat)
    fieldnames = sorted({key for row in flat_rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def copy_examples(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> dict[str, str]:
    examples_dir = output_dir / "examples"
    copied = {}
    for status, wanted_success in (("success", True), ("failure", False)):
        row = next((item for item in rollout_rows if item["success"] is wanted_success), None)
        if not row:
            continue
        target = examples_dir / status
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(row["rollout_dir"], target)
        copied[status] = str(target)
    return copied


def main() -> None:
    args = parse_args()
    env_cfg = default_env_cfg(args.seed)
    env_cfg.update({"robot": args.robot, "task": args.task, "layout": args.layout, "device": args.device})
    env_cfg = deep_merge(env_cfg, parse_overrides(args.overrides).get("env_cfg", {}))

    from lw_benchhub.distributed.proxy import RemoteEnv

    env = RemoteEnv.make(address=(args.ipc_host, args.ipc_port), authkey=args.ipc_authkey.encode())
    env.attach(SimpleNamespace(**env_cfg))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump({"env_cfg": to_jsonable(env_cfg), "benchmark_args": vars(args)}, f, sort_keys=False)

    rollout_rows: list[dict[str, Any]] = []
    success_count = 0
    for episode_index in range(args.episodes):
        rollout_dir = output_dir / "rollouts" / f"test_{episode_index:03d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        video_path = rollout_dir / "record_video.mp4"

        obs, _ = env.reset()
        init_images = write_obs_images(obs, args.record_camera, rollout_dir / "init_images", env)
        success = False
        last_extras: dict[str, Any] = {}
        steps = 0
        import mediapy as media

        with media.VideoWriter(
            path=video_path,
            shape=(args.height, args.width * max(1, len(args.record_camera))),
            fps=30,
        ) as writer:
            first_frame = combine_camera_frame(obs, args.record_camera, args.height, args.width, env)
            if first_frame is not None:
                writer.add_image(first_frame)
            for steps in range(1, args.max_steps + 1):
                action = action_for_space(env, args.action_mode)
                obs, _, terminated, _, extras = env.step(action)
                last_extras = to_jsonable(extras)
                frame = combine_camera_frame(obs, args.record_camera, args.height, args.width, env)
                if frame is not None:
                    writer.add_image(frame)
                success = success_from_step(terminated, extras)
                import torch

                if success or bool(torch.as_tensor(terminated).any().item()):
                    break

        if success:
            success_count += 1
        end_images = write_obs_images(obs, args.record_camera, rollout_dir / "end_images", env)
        rollout_meta = {
            "episode_index": episode_index,
            "success": success,
            "time_steps": steps,
            "video_path": str(video_path),
            "init_images": init_images,
            "end_images": end_images,
            "extras": last_extras,
            "checker_results": checker_results(env),
            "task": env_cfg.get("task"),
            "robot": env_cfg.get("robot"),
            "layout": env_cfg.get("layout"),
            "scene_backend": env_cfg.get("scene_backend"),
            "task_backend": env_cfg.get("task_backend"),
            "instruction": None,
            "action_mode": args.action_mode,
        }
        with open(rollout_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(rollout_meta), f, indent=2)
        rollout_rows.append({**rollout_meta, "rollout_dir": rollout_dir})
        print(f"Episode {episode_index}: success={success} steps={steps}")

    examples = copy_examples(output_dir, rollout_rows)
    summary = {
        "requested_test_count": args.episodes,
        "test_count": len(rollout_rows),
        "success_count": success_count,
        "failure_count": len(rollout_rows) - success_count,
        "success_rate": success_count / len(rollout_rows) if rollout_rows else 0.0,
        "output_dir": str(output_dir),
        "config_path": str(output_dir / "config.yaml"),
        "examples": examples,
        "rollouts": [{k: v for k, v in row.items() if k != "rollout_dir"} for row in rollout_rows],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    write_csv(output_dir / "metrics.csv", summary["rollouts"])

    env.close()
    env.close_connection()


if __name__ == "__main__":
    main()
