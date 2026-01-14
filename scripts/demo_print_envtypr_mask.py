# -*- coding: utf-8 -*-
"""
Demo: reset once and print env_type + act_mask (first 10 envs)

Run:
  python scripts/demo_print_envtype_mask.py --task <你的task> --num_envs 64
"""

import argparse
import sys

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()

    # keep hydra args clean (same pattern as IsaacLab scripts)
    sys.argv = [sys.argv[0]] + hydra_args

    # 1) launch isaac sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 2) import after app launch
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import multi_loco.tasks  # noqa: F401

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    # ✅ 你的 mdp（里面有 act_mask 函数）
    from multi_loco.tasks.manager_based.multi_loco import mdp

    # 3) load env cfg from registry
    env_cfg = parse_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device

    # 4) make env
    env = gym.make(args.task, cfg=env_cfg)

    # 5) reset once
    obs, info = env.reset()


    # 6) print env_type and act_mask
    base_env = env.unwrapped  # ManagerBasedRLEnv

    a = torch.randn((base_env.num_envs, 12), device=base_env.device)
    a_masked = a * base_env.act_mask
    print("max masked tail for biped:", a_masked[base_env.env_type==0, 6:].abs().max().item())


    env_unwrapped = env.unwrapped
    print("env_type unique:", torch.unique(env_unwrapped.env_type))
    print("mask unique rows:", torch.unique(env_unwrapped.act_mask, dim=0))
    print("biped tail max:", env_unwrapped.act_mask[env_unwrapped.env_type==0, 6:].max().item())
    print("quad tail min:", env_unwrapped.act_mask[env_unwrapped.env_type==1, 6:].min().item())


    if hasattr(base_env, "env_type"):
        print("env.env_type[:10] =")
        print(base_env.env_type[:10].detach().cpu())
    else:
        print("env.env_type 不存在（说明 sample_env_type 还没在 reset event 里执行）")

    mask = mdp.act_mask(base_env, action_dim=12)  # (num_envs, 12)
    print("\nact_mask[:10] =")
    print(mask[:10].detach().cpu())

    # optional sanity: count biped/quad
    if hasattr(base_env, "env_type"):
        n_biped = int((base_env.env_type == 0).sum().item())
        n_quad = int((base_env.env_type == 1).sum().item())
        print(f"\ncount: biped={n_biped}, quad={n_quad}")




    # close
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
