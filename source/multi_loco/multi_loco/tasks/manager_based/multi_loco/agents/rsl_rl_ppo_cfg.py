# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,RslRlPpoActorCriticRecurrentCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    clip_actions = 18.0
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "multi_loco"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticMultiCritic_GNN",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )

@configclass
class MultiLocoRoughPPORunnerCfg(PPORunnerCfg):
    experiment_name = "multi_loco_rough"

@configclass
class MultiLocoFlatPPORunnerCfg(PPORunnerCfg):
    experiment_name = "multi_loco_flat"