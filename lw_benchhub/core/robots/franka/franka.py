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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.utils.pose import Pose
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

import lw_benchhub.core.mdp as mdp
from lw_benchhub.core.robots.robot_arena_base import EmbodimentBaseObservationCfg, EmbodimentBasePolicyObservationCfg, LwEmbodimentBase
from lw_benchhub.utils.env import ExecuteMode

##
# Pre-defined configs
##
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class FrankaActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.DifferentialInverseKinematicsActionCfg | mdp.JointPositionActionCfg = MISSING
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaPolicyObservationsCfg(EmbodimentBasePolicyObservationCfg):
    """Observations for policy group with state values."""

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class FrankaCameraCfg:
    external_camera: TiledCameraCfg = None
    global_camera: TiledCameraCfg = None
    hand_camera: TiledCameraCfg = None


@configclass
class FrankaSceneCfg:
    # Listens to the required transforms
    # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
    # the other frames are the fingers
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )


class FrankaEnvCfg(LwEmbodimentBase):
    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "franka"
        self.observation_config = EmbodimentBaseObservationCfg()
        self.policy_observation_config = FrankaPolicyObservationsCfg()
        self.action_config = FrankaActionsCfg()
        self.scene_config = FrankaSceneCfg()
        self.camera_config = FrankaCameraCfg()
        self.robot_scale = 1.0
        self.scene_config.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.observation_cameras = {
            "external_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/external_camera",
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(4.5, -1.0, 6.0),
                        rot=(-0.125, 0.362, 0.873, -0.302),
                        convention="ros",
                    ),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True,
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL],
            },
            "global_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/global_camera",
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(4.5, -1.0, 6.0),
                        rot=(-0.125, 0.362, 0.873, -0.302),
                        convention="ros",
                    ),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True,
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL],
            },
            "hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(0.05, 0.0, 0.0),
                        rot=(0.0, 0.707107, 0.707107, 0.0),
                        convention="opengl",
                    ),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=36.5,
                        focus_distance=400.0,
                        horizontal_aperture=36.83,
                        clipping_range=(0.01, 3.0),
                        lock_camera=True,
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL],
            },
        }


@configclass
class FrankaAbsActionsCfg(FrankaActionsCfg):
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )


class FrankaAbsEnvCfg(FrankaEnvCfg):
    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.action_config = FrankaAbsActionsCfg()
        self.scene_config.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class FrankaRLActionsCfg(FrankaActionsCfg):
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=1, use_default_offset=True
    )


class FrankaRLEnvCfg(FrankaEnvCfg):
    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.action_config = FrankaRLActionsCfg()
        self.scene_config.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.reward_gripper_joint_names = ["panda_joint.*"]
