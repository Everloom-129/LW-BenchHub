# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import csv
import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import mediapy as media
import numpy as np
import tqdm
import yaml

sys.path.append("./")
sys.path.append("../../policy")

# add argparse arguments
# Get project root directory (assuming script is in lw_benchhub/scripts/policy/)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
default_config_path = project_root / "policy" / "PI" / "deploy_policy_lerobot.yml"

parser = argparse.ArgumentParser(description="Eval policy in Isaac Lab environments.")
parser.add_argument("--config", type=str, default=str(default_config_path))
parser.add_argument("--overrides", nargs=argparse.REMAINDER)
parser.add_argument("--save_states", action="store_true", help="Save states after each step")
parser.add_argument("--output_dir", type=str, default="./results/policy_eval", help="Directory for benchmark artifacts")
parser.add_argument("--run_name", type=str, default=None, help="Optional run subdirectory name")

# parse the arguments
args_cli = parser.parse_args()


def parse_args_and_config():

    with open(args_cli.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]

            try:
                value = eval(value)
            except Exception:
                pass

            # use ':' to split config
            if ':' in key:
                keys = key.split(':')
                current_level = override_dict
                for k in keys[:-1]:
                    if k not in current_level:
                        current_level[k] = {}
                    current_level = current_level[k]
                current_level[keys[-1]] = value
            else:
                override_dict[key] = value

        return override_dict

    def deep_merge(original, update):
        for key, value in update.items():
            if (key in original and
                isinstance(original[key], dict) and
                    isinstance(value, dict)):
                deep_merge(original[key], value)
            else:
                original[key] = value
        return original

    if args_cli.overrides:
        overrides = parse_override_pairs(args_cli.overrides)
        config = deep_merge(config, overrides)

    return config


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _to_jsonable(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return repr(value)


def _camera_image_from_obs(obs, camera_name):
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


def _image_to_uint8(image):
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


def _render_scene(env):
    try:
        return _image_to_uint8(env.render())
    except Exception:
        return None


def _write_obs_images(obs, camera_names, image_dir, env=None):
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for camera_name in camera_names:
        image = _camera_image_from_obs(obs, camera_name)
        if image is None:
            continue
        image_path = image_dir / f"{camera_name}.png"
        media.write_image(image_path, image)
        paths[camera_name] = str(image_path)
    if env is not None:
        scene_image = _render_scene(env)
        if scene_image is not None:
            image_path = image_dir / "scene.png"
            media.write_image(image_path, scene_image)
            paths["scene"] = str(image_path)
    return paths


def _try_get_checker_results(env):
    try:
        return _to_jsonable(env._svc.call("unwrapped.cfg.isaaclab_arena_env.task.get_checker_results"))
    except Exception as exc:
        return {"_error": f"checker_results_unavailable: {exc}"}


def _flatten_metrics(prefix, value, output):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(child_prefix, child, output)
    elif isinstance(value, list):
        if len(value) <= 8 and all(not isinstance(item, (dict, list)) for item in value):
            output[prefix] = value
        else:
            output[prefix] = json.dumps(value)
    else:
        output[prefix] = value


def _write_metrics_csv(path, rows):
    flat_rows = []
    for row in rows:
        flat = {}
        _flatten_metrics("", row, flat)
        flat_rows.append(flat)
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def _copy_example_rollouts(output_dir, rollout_rows):
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


def main(usr_args):

    from lw_benchhub.distributed.proxy import RemoteEnv
    env = RemoteEnv.make(address=('127.0.0.1', 50000), authkey=b'lightwheel')
    if "env_cfg" in usr_args and usr_args["env_cfg"]:
        env_cfg = dict(usr_args["env_cfg"])
        defaults = {
            "scene_backend": "robocasa",
            "task_backend": "robocasa",
            "device": "cuda:0",
            "rl": None,
            "robot_scale": 1.0,
            "first_person_view": False,
            "disable_fabric": False,
            "num_envs": 1,
            "usd_simplify": False,
            "video": False,
            "for_rl": False,
            "variant": "Visual",
            "concatenate_terms": False,
            "distributed": False,
            "seed": 42,
            "sources": None,
            "object_projects": None,
            "execute_mode": "eval",
            "replay_cfgs": {"add_camera_to_observation": True},
        }
        for key, value in defaults.items():
            if key not in env_cfg:
                env_cfg[key] = value
        env_cfg = SimpleNamespace(**env_cfg)
    env.attach(env_cfg)

    policy_name = usr_args["policy_name"]
    policy_module = importlib.import_module("policy")
    policy_class = getattr(policy_module, policy_name)
    policy = policy_class(usr_args)

    usr_args['actions_dim'] = env.action_space.shape[1]
    usr_args['decimation'] = env.unwrapped.cfg.decimation

    output_dir = Path(args_cli.output_dir)
    if args_cli.run_name:
        output_dir = output_dir / args_cli.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(exist_ok=True)

    config_snapshot_path = output_dir / "config.yaml"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_jsonable(usr_args), f, sort_keys=False)

    test_num = usr_args.get('test_num', 10)  # default 10
    suc_num = 0
    rollout_rows = []
    with (
        contextlib.suppress(KeyboardInterrupt),  # and torch.inference_mode(),
    ):
        for idx in tqdm.tqdm(range(test_num)):
            rollout_dir = output_dir / "rollouts" / f"test_{idx:03d}"
            eval_video_path = rollout_dir / "record_video.mp4"
            eval_video_path.parent.mkdir(parents=True, exist_ok=True)

            usr_args['save_path'] = str(rollout_dir)
            usr_args["_rollout_step_count"] = 0
            usr_args["_last_observation"] = None
            usr_args["_last_extras"] = {}
            with media.VideoWriter(path=eval_video_path, shape=(usr_args['height'], usr_args['width'] * len(usr_args['record_camera'])), fps=30) as v:
                obs, _ = env.reset()
                init_image_paths = _write_obs_images(obs, usr_args['record_camera'], rollout_dir / "init_images", env)
                policy.reset_model()
                has_success = policy.eval(env, obs, usr_args, v)
                end_obs = usr_args.get("_last_observation") or obs
                end_image_paths = _write_obs_images(end_obs, usr_args['record_camera'], rollout_dir / "end_images", env)
                if has_success:
                    suc_num += 1
                checker_results = _try_get_checker_results(env)
                extras = _to_jsonable(usr_args.get("_last_extras", {}))
                rollout_meta = {
                    "episode_index": idx,
                    "success": bool(has_success),
                    "time_steps": int(usr_args.get("_rollout_step_count", 0)),
                    "video_path": str(eval_video_path),
                    "init_images": init_image_paths,
                    "end_images": end_image_paths,
                    "extras": extras,
                    "checker_results": checker_results,
                    "task": usr_args.get("env_cfg", {}).get("task"),
                    "robot": usr_args.get("env_cfg", {}).get("robot"),
                    "layout": usr_args.get("env_cfg", {}).get("layout"),
                    "scene_backend": usr_args.get("env_cfg", {}).get("scene_backend"),
                    "task_backend": usr_args.get("env_cfg", {}).get("task_backend"),
                    "instruction": usr_args.get("instruction"),
                }
                with open(rollout_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(_to_jsonable(rollout_meta), f, indent=2, ensure_ascii=False)
                rollout_rows.append({**rollout_meta, "rollout_dir": rollout_dir})
                print(f"Current test result: {has_success}. Success/total tested: {suc_num}/{idx+1}")
    actual_count = len(rollout_rows)
    success_rate = suc_num / actual_count if actual_count else 0.0
    print(f"Success rate: {success_rate}")

    examples = _copy_example_rollouts(output_dir, rollout_rows)
    results = {
        "requested_test_count": test_num,
        "test_count": actual_count,
        "success_count": suc_num,
        "failure_count": actual_count - suc_num,
        "success_rate": success_rate,
        "output_dir": str(output_dir),
        "config_path": str(config_snapshot_path),
        "examples": examples,
        "rollouts": [{k: v for k, v in row.items() if k != "rollout_dir"} for row in rollout_rows],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    _write_metrics_csv(output_dir / "metrics.csv", results["rollouts"])

    env.close()
    # env.detach()
    env.close_connection()


if __name__ == "__main__":
    # example: "python lw_benchhub/scripts/policy/eval_policy.py --config policy/GR00T/deploy_policy_piper.yml \
    #           --overrides --env_cfg:task SizeSorting --env_cfg:layout robocasakitchen \
    #           --instruction  "Stack objects on counter from large to small" --test_num 10
    # run the main function
    usr_args = parse_args_and_config()
    main(usr_args)
