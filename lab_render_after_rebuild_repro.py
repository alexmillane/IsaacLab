# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab-only reproduction of the "renders wrong after a stage rebuild" bug.

Builds a stock camera-carrying Lab task twice in one process, performing a full stage teardown
between builds, and compares the two builds' camera images. The default task carries a wrist camera
parented to the robot's hand and a table camera parented to the environment.

With Fabric on, the rebuilt stage renders geometry at stale poses: the Franka's hand and finger
meshes are mangled, and in the worst runs the table surface goes dark and camera geometry appears at
the world origin. With Fabric off the two builds match. The script exits non-zero when a rebuild
renders differently.

The damage is nondeterministic -- across repeated runs it has ranged from 0% to 81% of pixels per
camera -- so a single run understates it. Use ``--builds`` to rebuild more than once per run.

.. code-block:: bash

    # Fabric on (the configuration that reproduces)
    uv run python lab_render_after_rebuild_repro.py --use_fabric 1 --out repro_out

    # Fabric off (control: expected to pass)
    uv run python lab_render_after_rebuild_repro.py --use_fabric 0 --out repro_out_nofabric
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal Lab-only post-rebuild render repro.")
parser.add_argument("--task", type=str, default="IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor")
parser.add_argument("--use_fabric", type=int, default=1, help="1 = Fabric on (default), 0 = Fabric off.")
parser.add_argument("--steps", type=int, default=30, help="Environment steps per build before capturing.")
parser.add_argument("--builds", type=int, default=2, help="Number of build + capture cycles.")
parser.add_argument("--out", type=str, default="", help="Directory to save the compared frames into, if given.")
AppLauncher.add_app_launcher_args(parser)
# Camera sensors require the rendering pipeline, and the comparison is offscreen. Neither is exposed
# as a CLI flag, but both are honoured as launcher-arg keys, so they are set as parser defaults.
parser.set_defaults(enable_cameras=True, headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import gymnasium as gym
import torch

import omni.timeline
import omni.usd

from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401  (registers the Lab tasks)
from isaaclab_tasks.utils import parse_env_cfg

# A pixel counts as changed above this per-channel 0-255 difference, and this fraction of changed
# pixels is tolerated. Measured over six runs on one machine (RTX 5090, cuda:0) with the default
# task: Fabric off stays at 0.0-0.1% across the rebuild, while Fabric on ranges from 0.0% to 80.7%
# depending on the run.
PIXEL_DIFFERENCE_TOLERANCE = 8
MAX_CHANGED_PIXEL_FRACTION = 0.02
# Minimum per-image standard deviation, so a pair of blank renders cannot pass the comparison
# vacuously -- a rebuild that renders nothing at all must fail rather than match.
MIN_IMAGE_STD = 1.0


def build_and_capture() -> dict[str, torch.Tensor]:
    """Build the task, settle it with zero actions, and return each camera's RGB image."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=bool(args_cli.use_fabric))
    # Same seed in both builds, so reset randomization draws the same scene.
    env_cfg.seed = 0
    # Without this a reset can return while assets are still streaming, which blanks or misplaces
    # geometry in the first frames and would be misread as a rebuild failure.
    env_cfg.wait_for_textures = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args_cli.steps):
            obs = env.step(actions)[0]
        # The image observation terms are the 4-dimensional ones: (num_envs, height, width, channels).
        # ``PolicyCfg`` sets ``concatenate_terms=False``, so ``obs["policy"]`` is keyed by term name.
        images = {name: value[0].detach().cpu().clone() for name, value in obs["policy"].items() if value.ndim == 4}
    finally:
        teardown_stage(env)
    return images


def teardown_stage(env) -> None:
    """Tear the stage down between builds, leaving the app running."""
    # Closing the environment also clears the SimulationContext singleton.
    env.close()
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    omni.timeline.get_timeline_interface().stop()
    omni.usd.get_context().new_stage()


def save_images(images: dict[str, torch.Tensor], tag: str) -> None:
    """Write one PNG per camera into ``--out``."""
    from PIL import Image

    os.makedirs(args_cli.out, exist_ok=True)
    for name, image in images.items():
        Image.fromarray(image[..., :3].numpy()).save(os.path.join(args_cli.out, f"{name}-{tag}.png"))


def save_difference_images(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], tag: str) -> None:
    """Write the absolute difference between two builds' renders, one PNG per camera."""
    from PIL import Image

    os.makedirs(args_cli.out, exist_ok=True)
    for name, image in candidate.items():
        difference = (reference[name][..., :3].float() - image[..., :3].float()).abs().to(torch.uint8)
        Image.fromarray(difference.numpy()).save(os.path.join(args_cli.out, f"{name}-{tag}-difference.png"))


def compare(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], tag: str) -> bool:
    """Report per-camera differences against the first build and return whether they are tolerable."""
    within_tolerance = True
    for name, before in reference.items():
        after = candidate[name]
        image_std = float(before.float().std())
        difference = (before.float() - after.float()).abs()
        changed_fraction = float(difference.gt(PIXEL_DIFFERENCE_TOLERANCE).float().mean())
        camera_ok = image_std > MIN_IMAGE_STD and changed_fraction <= MAX_CHANGED_PIXEL_FRACTION
        within_tolerance &= camera_ok
        print(
            f"[repro] {tag} {name}: {changed_fraction:.1%} of pixels changed by more than"
            f" {PIXEL_DIFFERENCE_TOLERANCE}/255, build0 image std {image_std:.3f}"
            f" -> {'OK' if camera_ok else 'CHANGED'}",
            flush=True,
        )
    return within_tolerance


def main() -> int:
    """Build the task across stage rebuilds and report how much each camera changed."""
    print(
        f"[repro] task={args_cli.task} device={args_cli.device}"
        f" use_fabric={bool(args_cli.use_fabric)} builds={args_cli.builds}",
        flush=True,
    )
    reference: dict[str, torch.Tensor] = {}
    within_tolerance = True
    for build_index in range(args_cli.builds):
        tag = f"build{build_index}"
        images = build_and_capture()
        if args_cli.out:
            save_images(images, tag)
        if build_index == 0:
            reference = images
            continue
        if args_cli.out:
            save_difference_images(reference, images, tag)
        within_tolerance &= compare(reference, images, tag)

    if within_tolerance:
        print("[repro] PASS: every rebuild rendered the same scene as the first build.", flush=True)
    else:
        print(
            "[repro] FAIL: a rebuild rendered differently from the first build. "
            "Scene geometry likely failed to render into the rebuilt stage.",
            flush=True,
        )
    return 0 if within_tolerance else 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        # Isaac Sim's shutdown ends the process with exit code 0, so a failure has to exit first for
        # the exit code to survive (which `git bisect run` depends on).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    simulation_app.close()
    sys.exit(exit_code)
