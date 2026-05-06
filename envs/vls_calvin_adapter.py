"""LangSteer-compatible adapter for VLS evaluation on CALVIN.

Subclasses VLS's CalvinAdapter to add:
  - Deterministic starting condition injection (for fair comparison with LangSteer baseline)
  - CALVIN task-oracle success detection (consistent with CalvinEnvironment.step())

Usage in run_vls_evaluation.py:
    adapter = LangSteerCalvinAdapter(
        raw_calvin_env=gym_wrapper._env,
        env_config={"vlm_camera": "static", "max_episode_steps": 360},
        task_oracle=task_oracle,
        device="cuda",
    )
    adapter.stage_starting_condition(robot_obs, scene_obs, task_name="open_drawer",
                                     instruction="open the drawer")
    obs, info = adapter.reset()
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

VLS_ROOT = Path(__file__).parent.parent / "third_party" / "vls"


def _load_vls_calvin_adapter():
    """Load VLS's CalvinAdapter under a 'vls.*' namespace to avoid conflicting
    with LangSteer's own 'core' package that may already be in sys.modules."""
    pkg = "vls.core.env_adapters"

    if pkg + ".calvin_adapter" in sys.modules:
        return sys.modules[pkg + ".calvin_adapter"].CalvinAdapter

    # Register package stubs so relative imports inside the VLS files resolve.
    for mod_name, search_path in [
        ("vls", str(VLS_ROOT)),
        ("vls.core", str(VLS_ROOT / "core")),
        ("vls.core.env_adapters", str(VLS_ROOT / "core" / "env_adapters")),
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = [search_path]
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub

    def _exec_module(full_name, rel_path, package):
        if full_name in sys.modules:
            return sys.modules[full_name]
        spec = importlib.util.spec_from_file_location(
            full_name, VLS_ROOT / rel_path
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = package
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _exec_module(
        "vls.core.env_adapters.base_adapter",
        "core/env_adapters/base_adapter.py",
        "vls.core.env_adapters",
    )
    calvin_mod = _exec_module(
        "vls.core.env_adapters.calvin_adapter",
        "core/env_adapters/calvin_adapter.py",
        "vls.core.env_adapters",
    )
    return calvin_mod.CalvinAdapter


CalvinAdapter = _load_vls_calvin_adapter()


class LangSteerCalvinAdapter(CalvinAdapter):
    """CalvinAdapter extended with deterministic starting conditions and task-oracle success."""

    def __init__(self, raw_calvin_env, env_config: dict, task_oracle, device: str = "cuda"):
        super().__init__(raw_calvin_env, env_config, device)
        self._task_oracle = task_oracle
        self._start_info = None
        self._current_task_name: str = ""
        self._staged_robot_obs: Optional[np.ndarray] = None
        self._staged_scene_obs: Optional[np.ndarray] = None

        # _robot_base_pose is never set by VLS's CalvinAdapter (bug). Initialise
        # it from PyBullet using the robot body's actual base position/orientation.
        self._robot_base_pose = self._get_robot_base_pose_from_pybullet()

    def _get_robot_base_pose_from_pybullet(self):
        """Query PyBullet for the robot base pose and return a VLS Pose3D."""
        import pybullet as p
        Pose3D = sys.modules["vls.core.env_adapters.base_adapter"].Pose3D

        robot = self._env.robot
        cid = robot.cid
        pos, orn_xyzw = p.getBasePositionAndOrientation(robot.robot_uid, physicsClientId=cid)
        # PyBullet returns quaternion as [x, y, z, w]; Pose3D expects [w, x, y, z]
        orn_wxyz = [orn_xyzw[3], orn_xyzw[0], orn_xyzw[1], orn_xyzw[2]]
        return Pose3D(position=np.array(pos, dtype=np.float32),
                      quaternion=np.array(orn_wxyz, dtype=np.float32))

    def stage_starting_condition(
        self,
        robot_obs: np.ndarray,
        scene_obs: np.ndarray,
        task_name: str,
        instruction: str = "",
    ) -> None:
        """Call before each episode to inject pre-sampled start state and set task."""
        self._staged_robot_obs = robot_obs
        self._staged_scene_obs = scene_obs
        self._current_task_name = task_name
        self._env_config["instruction"] = instruction

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        """Reset with staged (robot_obs, scene_obs) if available, then capture start_info."""
        robot_obs = self._staged_robot_obs
        scene_obs = self._staged_scene_obs

        if robot_obs is not None or scene_obs is not None:
            obs = self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        else:
            obs = self._env.reset()

        self.update_state_dict(obs)
        self.init_scene_state_dict = None
        self.episode_step = 0
        self._staged_robot_obs = None
        self._staged_scene_obs = None

        try:
            self._start_info = self._env.get_info()
        except Exception:
            self._start_info = None

        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def check_success(self) -> Tuple[bool, str]:
        """Use CALVIN task oracle (same as CalvinEnvironment) for consistent success detection."""
        if self._start_info is None or not self._current_task_name:
            return False, "no_behavior"

        try:
            end_info = self._env.get_info()
            achieved = self._task_oracle.get_task_info_for_set(
                self._start_info, end_info, {self._current_task_name}
            )
            if self._current_task_name in achieved:
                return True, self._current_task_name
        except Exception:
            pass

        return False, "no_behavior"

    def get_keypoint_detection_inputs(self):
        """Render RGB, depth, and segmentation via PyBullet at 640×640.

        VLS's CalvinAdapter calls render(mode="rgb+depth+segmentation", height=..., width=...)
        which is only supported by the Treeeplanter/calvin_env fork, not the standard build
        in our venv.  We replicate the same output using pybullet.getCameraImage() directly,
        then delegate segmentation processing to the parent class.
        """
        import pybullet as p

        W, H = 640, 640
        cam = self._env.cameras[0]  # static camera

        proj_matrix = p.computeProjectionMatrixFOV(
            cam.fov, W / H, cam.nearval, cam.farval,
            physicsClientId=cam.cid,
        )
        _, _, rgba, depth_buf, seg_raw = p.getCameraImage(
            W, H, cam.viewMatrix, proj_matrix,
            physicsClientId=cam.cid,
        )

        rgb = np.array(rgba, dtype=np.uint8)[:, :, :3]

        # Convert normalised depth buffer [0,1] → metres
        near, far = cam.nearval, cam.farval
        depth = far * near / (far - (far - near) * np.array(depth_buf, dtype=np.float32))

        seg_raw = np.array(seg_raw, dtype=np.int32).reshape(H, W)

        segmentation, _, segment_id_to_name = self.process_segmentation(seg_raw)
        camera_params = self.get_camera_params(self.vlm_camera)
        points = self._depth_to_pointcloud(depth, camera_params)

        return rgb, depth, points, segmentation, segment_id_to_name

    def get_vlm_image(self):
        """Return a 640×640 RGB image from the static camera via PyBullet."""
        import pybullet as p

        W, H = 640, 640
        cam = self._env.cameras[0]
        proj_matrix = p.computeProjectionMatrixFOV(
            cam.fov, W / H, cam.nearval, cam.farval,
            physicsClientId=cam.cid,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            W, H, cam.viewMatrix, proj_matrix,
            physicsClientId=cam.cid,
        )
        return np.array(rgba, dtype=np.uint8)[:, :, :3]

    def get_instruction(self) -> str:
        return self._env_config.get("instruction", "")

    def get_task_description(self) -> str:
        return self.get_instruction()
