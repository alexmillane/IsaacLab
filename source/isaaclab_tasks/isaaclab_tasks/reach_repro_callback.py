# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEMP repro for the ArticulationCfg lazy_export duplication (IsaacLab-only).

Mimics the ``--external_callback`` flow that surfaces the bug: launch the app, then build the env
cfg *instance* right away (capturing the robot's ArticulationCfg class early) and register it via
``env_cfg_entry_point=<instance>``. train.py's later ``gym.make`` builds the InteractiveScene at a
different time, by which point ``articulation_cfg`` has been re-executed under lazy_export, so
``isinstance(robot, ArticulationCfg)`` is False and the scene raises "Unknown asset config type".

Run:
  train.py --task Isaac-Reach-Repro-v0 \\
    --external_callback isaaclab_tasks.reach_repro_callback.reach_repro_callback \\
    --num_envs 1 --max_iterations 1 --headless
"""


def app_only_callback():
    """Start the app early (like Arena) but let hydra build the env normally (string entry_point).

    Discriminates: if ``Isaac-Reach-Franka-v0`` still trains with this, then starting the app early
    is NOT the cause -- the early *env-cfg construction* is.
    """
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args, remaining = parser.parse_known_args()
    AppLauncher(app_args)
    return remaining


def reach_repro_callback():
    import argparse

    import gymnasium as gym

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args, remaining = parser.parse_known_args()
    AppLauncher(app_args)

    from isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents.rsl_rl_ppo_cfg import (
        FrankaReachPPORunnerCfg,
    )
    from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg

    # Robot ArticulationCfg captured HERE (early), before hydra_task_config / gym.make.
    env_cfg = FrankaReachEnvCfg()

    gym.register(
        id="Isaac-Reach-Repro-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": FrankaReachPPORunnerCfg(),
        },
    )
    return remaining
