# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .multi_loco_env import MultiLocoEnv

##
# Register Gym environments.
##


# gym.register(
#     id="Multi_Loco_v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.multi_loco_env_cfg:MultiLocoEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )

gym.register(
    id="Multi_Loco_Flat",
    entry_point=f"{__name__}.multi_loco_env:MultiLocoEnv",
    # entry_point= MultiLocoEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_loco_env_cfg:MultiLocoFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiLocoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Multi_Loco_Flat_Play",
    entry_point=f"{__name__}.multi_loco_env:MultiLocoEnv",
    # entry_point= MultiLocoEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_loco_env_cfg:MultiLocoFlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiLocoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Multi_Loco_Rough",
    entry_point=f"{__name__}.multi_loco_env:MultiLocoEnv",
    # entry_point= MultiLocoEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_loco_env_cfg:MultiLocoRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiLocoRoughPPORunnerCfg",
    },
)

gym.register(
    id="Multi_Loco_Rough_Play",
    entry_point=f"{__name__}.multi_loco_env:MultiLocoEnv",
    # entry_point= MultiLocoEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_loco_env_cfg:MultiLocoRoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiLocoRoughPPORunnerCfg",
    },
)