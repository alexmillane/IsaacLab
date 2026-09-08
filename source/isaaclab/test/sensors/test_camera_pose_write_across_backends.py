# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera pose writes must take effect on every physics backend.

:meth:`Camera.set_world_poses` writes through the sensor's :class:`FrameView`. Under PhysX that view is
Fabric-backed (:class:`FabricFrameView`), so the RTX renderer -- which reads the USD/Fabric camera prim --
follows the write. Under Newton the view is a :class:`NewtonSiteFrameView`, which updates only in-memory
Warp state; nothing mirrors the pose back to the camera prim, so the rendered image keeps the old pose
even though ``camera.data.pos_w`` reports the new one.

Both tests move a downward-looking camera straight up over a ground plane, which multiplies the distance
to every visible surface, and check the two observable consequences of the write:

- ``_moves_reported_pose_*`` reads ``camera.data.pos_w`` (deterministic, no renderer involved).
- ``_moves_render_*`` compares the rendered depth before and after the move.

:func:`test_camera_pose_update_reflected_in_render` in ``test_camera.py`` covers the render half of this on
the default (PhysX) backend only; these tests parametrize the backend so the Newton regression is visible.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

BACKEND_CFGS = [PhysxCfg(), NewtonCfg(solver_cfg=MJWarpSolverCfg())]
BACKEND_IDS = ["physx", "newton"]

DEVICE = "cuda:0"
HEIGHT = 128
WIDTH = 256
# Camera heights [m] above the ground plane, looking straight down. Moving between them multiplies the
# distance to every visible surface, so both the reported pose and the rendered depth must change.
CAMERA_HEIGHT_CLOSE_M = 2.0
CAMERA_HEIGHT_FAR_M = 8.0
# Reported world-pose shift [m] below which the write is considered to have been dropped.
POSE_SHIFT_THRESHOLD_M = 0.5 * (CAMERA_HEIGHT_FAR_M - CAMERA_HEIGHT_CLOSE_M)
# Minimum ratio of far-to-close mean depth. The true ratio is ~4x; 1.5 leaves room for the ground plane
# filling different fractions of the frame while still failing hard if the render does not move at all.
DEPTH_RATIO_THRESHOLD = 1.5
# Physics steps taken after each pose write so the renderer produces a frame at the new pose.
STEPS_PER_POSE = 2


@configclass
class _SceneCfg(InteractiveSceneCfg):
    """A single rigid body under the camera; Newton cannot build a model from an empty scene."""

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )


def _camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="/World/Camera",
        height=HEIGHT,
        width=WIDTH,
        update_period=0,
        update_latest_camera_pose=True,
        data_types=["distance_to_camera"],
        # Spawn already looking straight down from the close height, so a dropped pose write leaves a
        # valid (but unchanged) depth image rather than an empty one.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, CAMERA_HEIGHT_CLOSE_M), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )


def _capture_at_heights(physics_cfg) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Move the camera to each height in turn, returning ``(pos_w, depth)`` captured at each one.

    The camera looks straight down (identity orientation in the OpenGL convention) at the ground plane, so
    the depth it reports is dominated by its height.
    """
    sim_cfg = SimulationCfg(physics=physics_cfg, device=DEVICE)
    captures = []
    with build_simulation_context(device=DEVICE, sim_cfg=sim_cfg, add_ground_plane=True, add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.0))
        camera = Camera(_camera_cfg())
        sim.reset()

        # OpenGL convention: the camera looks along its -Z axis, so identity orientation looks straight down.
        orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
        for height in (CAMERA_HEIGHT_CLOSE_M, CAMERA_HEIGHT_FAR_M):
            positions = torch.tensor([[0.0, 0.0, height]], device=DEVICE)
            camera.set_world_poses(positions, orientations, convention="opengl")
            for _ in range(STEPS_PER_POSE):
                sim.step()
            camera.update(sim.get_physics_dt())
            depth = camera.data.output["distance_to_camera"].torch.detach().float().cpu().clone()
            captures.append((camera.data.pos_w.torch.detach().float().cpu().clone(), depth))

        del camera
    return captures


def _mean_valid_depth(depth: torch.Tensor) -> float:
    """Mean depth over pixels that hit geometry (the sky renders at the far clipping range)."""
    valid = depth[torch.isfinite(depth) & (depth < _camera_cfg().spawn.clipping_range[1])]
    assert valid.numel() > 0, "No valid depth pixels; the camera sees no geometry."
    return valid.mean().item()


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_reported_pose(physics_cfg):
    """``camera.data.pos_w`` follows a ``set_world_poses`` write on every backend."""
    captures = _capture_at_heights(physics_cfg)

    pos_close, pos_far = captures[0][0], captures[1][0]
    np.testing.assert_allclose(pos_close.numpy(), [[0.0, 0.0, CAMERA_HEIGHT_CLOSE_M]], atol=1e-3)
    np.testing.assert_allclose(pos_far.numpy(), [[0.0, 0.0, CAMERA_HEIGHT_FAR_M]], atol=1e-3)
    shift = (pos_far - pos_close).norm(dim=-1).max().item()
    assert shift > POSE_SHIFT_THRESHOLD_M, (
        f"Expected camera.data.pos_w to follow the pose write (> {POSE_SHIFT_THRESHOLD_M} m); got {shift:.4f} m."
    )


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_render(physics_cfg):
    """The rendered depth follows a ``set_world_poses`` write on every backend.

    Under Newton the ``NewtonSiteFrameView`` write never reaches the camera prim the RTX renderer reads,
    so the image stays at the old pose and the depth ratio collapses to ~1.
    """
    captures = _capture_at_heights(physics_cfg)

    mean_close = _mean_valid_depth(captures[0][1])
    mean_far = _mean_valid_depth(captures[1][1])
    ratio = mean_far / mean_close
    assert ratio > DEPTH_RATIO_THRESHOLD, (
        f"Far depth ({mean_far:.2f} m) should be > {DEPTH_RATIO_THRESHOLD}x close depth ({mean_close:.2f} m); "
        f"got {ratio:.2f}x. The camera pose write is not reaching the renderer."
    )
