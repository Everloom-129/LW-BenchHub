"""Run LW-BenchHub rollouts against an OpenPI pi05 DROID policy server."""

from __future__ import annotations

import argparse
import json
import os
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

from lw_benchhub.scripts.policy.molmoact_droid_rollouts import (  # noqa: E402
    DEFAULT_INSTRUCTIONS,
    DEFAULT_TASKS,
    attach_task,
    checker_results,
    combine_frame,
    copy_examples,
    deep_merge,
    default_env_cfg,
    extract_state,
    first_available_image,
    parse_overrides,
    success_from_step,
    to_jsonable,
    write_csv,
    write_obs_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenPI pi05 DROID policy rollouts in LW-BenchHub.")
    parser.add_argument("--output_dir", default="./results/openpi_pi05_droid_eval")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--action_chunk_steps", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=8002)
    parser.add_argument("--record_camera", nargs="+", default=["global_camera_rgb", "external_camera_rgb", "hand_camera_rgb"])
    parser.add_argument("--external_camera", default="external_camera_rgb")
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


def map_openpi_action(action: np.ndarray, action_shape: tuple[int, ...]) -> np.ndarray:
    """Map DROID 7 joint velocity + gripper actions onto the local Panda-RL action space."""
    flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
    if flat_action.size:
        flat_action = np.clip(flat_action, -1.0, 1.0)
    if flat_action.size >= 8:
        flat_action[7] = 1.0 if flat_action[7] > 0.5 else 0.0

    target_size = int(np.prod(action_shape))
    mapped = np.zeros(target_size, dtype=np.float32)
    take = min(target_size, flat_action.shape[0])
    mapped[:take] = flat_action[:take]
    return mapped.reshape(action_shape)


def prepare_image(image: np.ndarray) -> np.ndarray:
    from openpi_client import image_tools

    image = image_tools.resize_with_pad(np.asarray(image, dtype=np.uint8), 224, 224)
    return image_tools.convert_to_uint8(image)


def infer_actions(policy, external: np.ndarray, wrist: np.ndarray, instruction: str, state: np.ndarray) -> tuple[np.ndarray, float]:
    request = {
        "observation/exterior_image_1_left": prepare_image(external),
        "observation/wrist_image_left": prepare_image(wrist),
        "observation/joint_position": state[:7].astype(np.float32),
        "observation/gripper_position": state[7:8].astype(np.float32),
        "prompt": instruction,
    }
    start = time.perf_counter()
    response = policy.infer(request)
    dt_ms = (time.perf_counter() - start) * 1000.0
    if "actions" not in response:
        raise KeyError(f"OpenPI response missing 'actions'; keys={sorted(response)}")
    return np.asarray(response["actions"], dtype=np.float32), dt_ms


def main() -> None:
    args = parse_args()
    env_overrides = parse_overrides(args.overrides).get("env_cfg", {})

    from openpi_client import websocket_client_policy

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    print(f"OpenPI server metadata: {policy.get_server_metadata()}")

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
                        external = first_available_image(obs, [args.external_camera, "global_camera_rgb", "scene"], env)
                        wrist = first_available_image(obs, [args.wrist_camera, args.external_camera, "scene"], env)
                        state = extract_state(obs)
                        actions, dt_ms = infer_actions(policy, external, wrist, instruction, state)
                        actions = actions.reshape(-1, actions.shape[-1])[: args.action_chunk_steps]
                        action_queue = [np.asarray(item, dtype=np.float32) for item in actions]
                        inference_calls += 1
                        inference_ms.append(dt_ms)
                    action = map_openpi_action(action_queue.pop(0), env.action_space.shape)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    obs, _, terminated, _, extras = env.step(torch.as_tensor(action, device=device))
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
                "policy": "openpi/pi05_droid",
                "policy_checkpoint": "gs://openpi-assets/checkpoints/pi05_droid",
                "policy_server": f"ws://{args.policy_host}:{args.policy_port}",
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

    print(f"Wrote OpenPI rollout summary to {output_dir / 'summary.json'}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
