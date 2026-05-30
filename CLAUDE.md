# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

LW-BenchHub is an Isaac Lab–Arena–based embodied-AI benchmark for kitchen manipulation and loco-manipulation across many robots. Almost everything runs *inside* a headless or windowed NVIDIA Isaac Sim (kit) process, so iteration is slow (first launch compiles shaders for minutes, then caches) and requires an RTX GPU.

## Environment & setup gotchas

Install is `bash ./install.sh` inside a `conda create -n lw_benchhub python=3.11` env. It installs torch (cu128), `isaacsim[all]==5.0.0`, the `third_party/IsaacLab-Arena` submodule (which pulls `IsaacLab` + `Isaac-GR00T`), and this package editable. The following are **not** handled by `install.sh` and bite on a fresh machine:

- **`isaaclab.sh --install` needs `cmake`**, which it tries to `sudo apt` install (fails non-interactively). Install it without sudo, and use **conda-forge `cmake=3.31`** — pip's cmake and cmake 4.x both break `egl_probe`/robomimic builds.
- **Run-time env vars are required** for any script that boots Isaac Sim: `OMNI_KIT_ACCEPT_EULA=YES` (else `import isaacsim` blocks on a stdin prompt) and `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` (else `import pinocchio` — pulled in transitively via `lw_benchhub.core.mdp.actions`— dies with `CXXABI_1.3.15 not found`).
- **`skrl` is pinned `>=1.4.3` with no upper bound**; a fresh install pulls skrl 2.x which breaks PPO model construction. Pin `skrl<2` (1.4.3).
- The GUI/teleop path additionally needs `warp-lang==1.8.0` and `git lfs pull` (robot/scene USDs under `lw_benchhub/data/assets/*.usd` are LFS pointers until pulled).
- The `cuDeviceGetUuid` Warp CUDA warning at startup is benign. On an 8 GB GPU, never run two camera/render Isaac jobs concurrently (OOM), and always close the sim app in a `finally` so a crash doesn't leak GPU memory.

## Common commands

All Python entry points below assume the conda env is active and the two env vars above are exported. Run from the repo root (configs and asset paths are resolved relative to it).

- **RL train / eval**: `bash train.sh` / `bash eval.sh`, or directly `python ./lw_benchhub/scripts/rl/train.py --task_config <name> --headless` and `.../rl/play.py --task_config <name>_play`. Override `--num_envs` / `--max_iterations` for quick smoke runs (8 GB fits ~2 envs of a kitchen scene). Checkpoints land in `lw_benchhub_logs/skrl/<task>/<run>/checkpoints/` (only after ~`checkpoint_interval` timesteps).
- **Teleoperation / data collection**: `python ./lw_benchhub/scripts/teleop/teleop_main.py --task_config <name>` (set `record: true` in the config to save HDF5). Most robots need VR (`vr-controller`/`vr-hand`); only `pandaomron` has a keyboard mapping (`KEYCONTROLLER_MAP`).
- **Replay**: `python ./lw_benchhub/scripts/teleop/replay_demos.py --dataset_file <hdf5> --enable_cameras` (state) or `replay_action_demo.py --replay_mode {action,joint_target}`.
- **Policy server–client** (decoupled IL/VLA eval): start `python ./lw_benchhub/scripts/env_server.py --headless --enable_cameras` (serves the env over IPC, default `127.0.0.1:50000`), then run a client/benchmark in another terminal (`./run_droid_franka_benchmark.sh`, `./run_policy_benchmark.sh`, or `lw_benchhub/scripts/policy/eval_policy.py`).
- **Policy benchmark + results dashboard**: against a running `env_server.py`, run a checkpoint-free rollout sweep with `./run_droid_franka_benchmark.sh results/droid_franka_eval --episodes 10 --max_steps 200` (wraps `scripts/policy/benchmark_rollouts.py`, which executes zero/random actions — no policy needed), or evaluate a real checkpoint with `./run_policy_benchmark.sh <config.yml> <output_dir> --overrides ...` (wraps `eval_policy.py`). Both export the same `results/<run>/` artifact layout: `summary.json`, `metrics.csv`, per-rollout `rollouts/test_NNN/` (video + init/end camera images), and copied success/failure `examples/`. View with `streamlit run lw_benchhub/scripts/policy/dashboard.py -- --results_dir ./results` (needs the optional `dashboard` extra: `pip install -e ".[dashboard]"`, pulls pandas+streamlit). Validate a finished run's artifacts with `python lw_benchhub/scripts/policy/validate_results.py ./results/<run>`. Camera observations come from `FrankaCameraCfg` (external/global/hand `TiledCameraCfg`s added to the Panda robot in `core/robots/franka/franka.py`).
- **G1 locomotion demo (no teleop device)**: `python ./lw_benchhub/scripts/demo/g1_loco_walk.py --out results/g1_loco_walk.mp4` — records the bundled `loco.onnx` policy walking.
- **Lint / format check** (what CI runs): `python ci_run/check_format.py <path>` — runs `autopep8 --diff` + `flake8`. There is no unit-test suite for the core package; CI validates via dataset replay (`ci_run/replay.sh`, task lists in `ci_tasks.txt` / `teleop_ci_tasks.txt`).

## Architecture: the compositional env model

A simulation run is **one Scene × one Robot(embodiment) × one Task × (optionally) one RL config**, selected by a single YAML and assembled at launch. Understanding this composition is the key to the codebase.

**Config resolution (`lw_benchhub/utils/config_loader.py`).** Every entry script takes `--task_config <stem>`. The loader scans `configs/**/*.{yml,yaml}` (plus paths advertised by the `config_search_path` entry point) into a flat `stem → path` map, supports `_base_:` inheritance, and merges the result into `args_cli`. So a config's `task`, `robot`, `layout`, `rl`, `num_envs`, etc. fields become CLI args. Configs live under `configs/{rl,data_collection,policy,common,...}`.

**Registries + env assembly (`lw_benchhub/utils/env.py`).** Scenes, robots, tasks, and RL configs each register themselves via `gym.register` with composed IDs (e.g. `Robocasa-Robot-G1-RL`, `Robocasa-Scene-Usd`, `Robocasa-Rl-<Name>`). They are discovered through the `[project.entry-points.lw_benchhub_modules]` table in `pyproject.toml` (`lw_benchhub.core.scenes`, `lw_benchhub.core.robots`, `lw_benchhub_tasks`, `lw_benchhub_rl`). `parse_env_cfg(scene_backend, task_backend, task_name, robot_name, scene_name, rl_name, ...)` uses `load_cfg_cls_from_registry(...)` to fetch each piece and compose an `IsaacLabArenaManagerBasedRLEnvCfg`; scripts then `gym.make("Robocasa-<task>-<robot>-v0", cfg=env_cfg)`. Scene USDs (kitchen floorplans + objects) are downloaded on demand by the **Lightwheel SDK** into `~/.cache/lightwheel_sdk/` (no auth needed).

**The four pluggable axes.**
- **Scene** — kitchen layout/style USD (`lw_benchhub/core/scenes`), `robocasakitchen-<layout>-<style>`.
- **Robot / embodiment** (`lw_benchhub/core/robots`) — each robot has multiple *control variants* that differ only in their action config: e.g. G1 has `G1-RL` (direct arm joint control), `G1-Controller-DecoupledWBC` (PINK IK arms + homie WBC legs, VR teleop), and `G1-Loco` (`loco.onnx` walking + diff-IK arms). Articulation/actuator/PD configs live in `core/robots/.../assets_cfg.py`.
- **Task** (`lw_benchhub_tasks`) — object placement, success checker, reward/MDP terms (Lightwheel-Robocasa + LIBERO suites).
- **RL config** (`lw_benchhub_rl`) — per-(robot,task) env wrapper + skrl/rsl-rl agent cfg.

**Action terms wrap low-level policies (`lw_benchhub/core/mdp/actions`).** This is where bundled controllers plug in. `LegPositionAction` runs `loco.onnx`/`squat.onnx` (configs in `core/mdp/configs/g1_loco.yaml`, ckpts in `core/mdp/ckpt/`); `G1DecoupledWBCAction` runs the homie WBC ONNX (`data/ckpts/nv_wbc_*`) plus PINK IK for the upper body. These ONNX policies are the only *bundled, ready-to-run* learned policies; RL manipulation checkpoints must be trained yourself. Importing `core.mdp.actions` pulls in pinocchio (hence the `LD_LIBRARY_PATH` requirement).

**Decoupled policy API (`policy/`, `lw_benchhub/distributed/`).** A server–client split: `env_server.py` hosts the simulation env behind an IPC (or RESTful) wrapper; manipulation policies (`policy/GR00T`, `policy/PI`) are thin clients that connect to their own policy servers/checkpoints (user-supplied, not bundled).

**`third_party/IsaacLab-Arena`** is the upstream Isaac Lab–Arena dependency (with `IsaacLab` and `Isaac-GR00T` submodules), kept as-is; `IsaacLabArenaEnvironment` is the object that ties embodiment + scene + task + orchestrator together.

## Other entry points

`lw_benchhub/scripts/autosim/` (automated demo generation), `scripts/policy/convert_hdf5_to_lerobot_dataset*.py` (HDF5 → LeRobot dataset), `scripts/policy/{openpi,molmoact}_droid_rollouts.py` (DROID-style VLA rollouts that emit the same dashboard artifact layout), `scripts/maniskill_ppo/` (ManiSkill PPO baselines), `lw_benchhub/sim2real/` and `interact_api.py` (real-robot bridge). LeRobot support is an optional extra: `pip install -e ".[lerobot]"`.
