"""Run LW-BenchHub rollouts against a MolmoAct2-DROID inference server.

The MolmoAct2-DROID server follows the DROID real-Franka wire format:
two RGB cameras, an 8-D state ``[q1..q7, gripper]``, and an ``(N, 8)``
absolute joint/gripper action chunk.  This bridge adapts that protocol to the
LW-BenchHub remote environment and writes the same result layout consumed by
``dashboard.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.append(str(project_root))


DEFAULT_TASKS = ["LiftObj", "OpenDrawer", "CloseDrawer"]
DEFAULT_INSTRUCTIONS = {
    "LiftObj": "Pick up the object and lift it.",
    "OpenDrawer": "Open the drawer.",
    "CloseDrawer": "Close the drawer.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MolmoAct2-DROID policy rollouts in LW-BenchHub.")
    parser.add_argument("--output_dir", default="./results/molmoact_droid_eval")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--action_chunk_steps", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server_url", default="http://127.0.0.1:8000")
    parser.add_argument("--record_camera", nargs="+", default=["hand_camera_rgb", "scene"])
    parser.add_argument("--external_camera", default="global_camera_rgb")
    parser.add_argument("--wrist_camera", default="hand_camera_rgb")
    parser.add_argument("--robot", default="Panda-RL")
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


def default_env_cfg(seed: int, task: str, robot: str, layout: str, device: str) -> dict[str, Any]:
    return {
        "task": task,
        "robot": robot,
        "layout": layout,
        "rl": None,
        "scene_backend": "robocasa",
        "task_backend": "robocasa",
        "device": device,
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
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def camera_image_from_obs(obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    if camera_name == "scene":
        return None
    policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
    matching_keys = [key for key in policy_obs if key.startswith(camera_name)]
    if not matching_keys:
        return None
    return image_to_uint8(policy_obs[matching_keys[0]])


def render_scene(env) -> np.ndarray | None:
    try:
        with contextlib.suppress(Exception):
            env.sim.set_camera_view(eye=[1.1, -5.2, 1.8], target=[2.2, -3.8, 0.9])
        return image_to_uint8(env.render())
    except Exception:
        return None


def first_available_image(obs: dict[str, Any], names: list[str], env=None) -> np.ndarray:
    for name in names:
        image = render_scene(env) if name == "scene" and env is not None else camera_image_from_obs(obs, name)
        if image is not None and image.size and image.max(initial=0) > 0 and float(image.std()) > 5.0:
            return image
    return np.zeros((480, 640, 3), dtype=np.uint8)


def write_obs_images(obs: dict[str, Any], camera_names: list[str], image_dir: Path, env=None) -> dict[str, str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    import mediapy as media

    for camera_name in camera_names:
        image = render_scene(env) if camera_name == "scene" and env is not None else camera_image_from_obs(obs, camera_name)
        if image is None:
            continue
        image_path = image_dir / f"{camera_name}.png"
        media.write_image(image_path, image)
        paths[camera_name] = str(image_path)
    return paths


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return image[y_idx][:, x_idx]


def combine_frame(obs: dict[str, Any], camera_names: list[str], height: int, width: int, env=None) -> np.ndarray:
    frames = []
    for camera_name in camera_names:
        image = render_scene(env) if camera_name == "scene" and env is not None else camera_image_from_obs(obs, camera_name)
        if image is None:
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
        else:
            frames.append(resize_nearest(image, height, width))
    if not frames:
        frames.append(np.zeros((height, width, 3), dtype=np.uint8))
    return np.concatenate(frames, axis=1)


def extract_state(obs: dict[str, Any]) -> np.ndarray:
    policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
    joint_pos = None
    for key in ("joint_pos", "joint_pos_rel"):
        if key in policy_obs:
            joint_pos = policy_obs[key]
            break
    if joint_pos is None:
        matches = [value for key, value in policy_obs.items() if "joint_pos" in key]
        if matches:
            joint_pos = matches[0]
    if joint_pos is None:
        return np.zeros(8, dtype=np.float32)
    state = np.asarray(to_jsonable(joint_pos), dtype=np.float32).reshape(-1)
    out = np.zeros(8, dtype=np.float32)
    take = min(8, state.shape[0])
    out[:take] = state[:take]
    return out


def map_droid_action(action: np.ndarray, action_shape: tuple[int, ...]) -> np.ndarray:
    flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
    target_size = int(np.prod(action_shape))
    mapped = np.zeros(target_size, dtype=np.float32)
    take = min(target_size, flat_action.shape[0])
    mapped[:take] = flat_action[:take]
    return mapped.reshape(action_shape)


def infer_actions(server_url: str, external: np.ndarray, wrist: np.ndarray, instruction: str, state: np.ndarray, num_steps: int) -> tuple[np.ndarray, float]:
    import json_numpy
    import requests

    json_numpy.patch()
    payload = {
        "external_cam": external,
        "wrist_cam": wrist,
        "instruction": instruction,
        "state": state.astype(np.float32),
        "num_steps": num_steps,
        "timestamp": time.time(),
    }
    response = requests.post(f"{server_url.rstrip('/')}/act", data=json_numpy.dumps(payload), timeout=120)
    response.raise_for_status()
    body = json_numpy.loads(response.text)
    return np.asarray(body["actions"], dtype=np.float32), float(body.get("dt_ms", 0.0))


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


def checker_results(env) -> dict[str, Any]:
    try:
        return to_jsonable(env._svc.call("unwrapped.cfg.isaaclab_arena_env.task.get_checker_results"))
    except Exception as exc:
        return {"_error": f"checker_results_unavailable: {exc}"}


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


def attach_task(env, env_cfg: dict[str, Any]) -> None:
    try:
        env.detach()
    except Exception:
        pass
    env.attach(SimpleNamespace(**env_cfg))


def main() -> None:
    args = parse_args()
    env_overrides = parse_overrides(args.overrides).get("env_cfg", {})

    import requests

    health = requests.get(f"{args.server_url.rstrip('/')}/act", timeout=10)
    health.raise_for_status()
    print(f"MolmoAct2 server health: {health.text}")

    from lw_benchhub.distributed.proxy import RemoteEnv

    env = RemoteEnv.make(address=(args.ipc_host, args.ipc_port), authkey=args.ipc_authkey.encode())

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(exist_ok=True)

    rollout_rows: list[dict[str, Any]] = []
    success_count = 0
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump({"benchmark_args": vars(args), "env_overrides": to_jsonable(env_overrides)}, f, sort_keys=False)

    global_index = 0
    for task in args.tasks:
        env_cfg = default_env_cfg(args.seed, task, args.robot, args.layout, args.device)
        env_cfg = deep_merge(env_cfg, env_overrides)
        attach_task(env, env_cfg)
        instruction = DEFAULT_INSTRUCTIONS.get(task, task)

        for episode_index in range(args.episodes_per_task):
            rollout_dir = output_dir / "rollouts" / f"test_{global_index:03d}_{task}"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            video_path = rollout_dir / "record_video.mp4"
            obs, _ = env.reset()
            init_images = write_obs_images(obs, args.record_camera, rollout_dir / "init_images", env)
            success = False
            last_extras: dict[str, Any] = {}
            steps = 0
            inference_calls = 0
            inference_ms: list[float] = []
            action_queue: list[np.ndarray] = []

            import mediapy as media
            import torch

            with media.VideoWriter(
                path=video_path,
                shape=(args.height, args.width * max(1, len(args.record_camera))),
                fps=args.fps,
            ) as writer:
                writer.add_image(combine_frame(obs, args.record_camera, args.height, args.width, env))
                for steps in range(1, args.max_steps + 1):
                    if not action_queue:
                        external = first_available_image(obs, [args.external_camera, "scene"], env)
                        wrist = first_available_image(obs, [args.wrist_camera, args.external_camera, "scene"], env)
                        state = extract_state(obs)
                        actions, dt_ms = infer_actions(
                            args.server_url, external, wrist, instruction, state, args.action_chunk_steps
                        )
                        action_queue = [np.asarray(item, dtype=np.float32) for item in actions.reshape(-1, actions.shape[-1])]
                        inference_calls += 1
                        inference_ms.append(dt_ms)
                    action = map_droid_action(action_queue.pop(0), env.action_space.shape)
                    obs, _, terminated, _, extras = env.step(torch.as_tensor(action, device="cuda" if torch.cuda.is_available() else "cpu"))
                    last_extras = to_jsonable(extras)
                    writer.add_image(combine_frame(obs, args.record_camera, args.height, args.width, env))
                    success = success_from_step(terminated, extras)
                    if success or bool(torch.as_tensor(terminated).any().item()):
                        break

            if success:
                success_count += 1
            end_images = write_obs_images(obs, args.record_camera, rollout_dir / "end_images", env)
            rollout_meta = {
                "episode_index": global_index,
                "task_episode_index": episode_index,
                "success": success,
                "time_steps": steps,
                "video_path": str(video_path),
                "init_images": init_images,
                "end_images": end_images,
                "extras": last_extras,
                "checker_results": checker_results(env),
                "task": task,
                "robot": env_cfg.get("robot"),
                "layout": env_cfg.get("layout"),
                "scene_backend": env_cfg.get("scene_backend"),
                "task_backend": env_cfg.get("task_backend"),
                "instruction": instruction,
                "policy": "allenai/MolmoAct2-DROID",
                "policy_server": args.server_url,
                "inference_calls": inference_calls,
                "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else 0.0,
            }
            with open(rollout_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(to_jsonable(rollout_meta), f, indent=2)
            rollout_rows.append({**rollout_meta, "rollout_dir": rollout_dir})
            print(f"{task} episode {episode_index}: success={success} steps={steps}")
            global_index += 1

    examples = copy_examples(output_dir, rollout_rows)
    summary = {
        "requested_test_count": len(args.tasks) * args.episodes_per_task,
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

    try:
        env.detach()
    except Exception:
        pass
    env.close_connection()


if __name__ == "__main__":
    main()
