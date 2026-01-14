# -*- coding: utf-8 -*-
"""
Demo: spawn biped + quad in each env using InteractiveScene only.
- No gym.make
- No ManagerBasedRLEnvCfg validation
- No actions/observations/rewards/terminations needed

Run:
  python scripts/demo_scene_spawn_two_robots.py --num_envs 2 --real-time
"""

import argparse
import time

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--real-time", action="store_true", default=False)
    # add Isaac Sim launcher args (device/headless/etc.)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    # 1) Launch Isaac Sim FIRST
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 2) Import IsaacLab modules AFTER app is running (omni.* available)
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    # ✅ 导入你的 SceneCfg（就是你刚改过、里面有 biped + quad 的那个）
    # 路径按你工程实际改：multi_loco_env_cfg.py 里定义的 MultiLocoSceneCfg
    from multi_loco.tasks.manager_based.multi_loco.multi_loco_env_cfg import MultiLocoSceneCfg

    # 3) Create simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # 4) Create scene (this will spawn biped+quad in each env)
    scene_cfg = MultiLocoSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # 5) Reset (writes initial states to sim)
    sim.reset()
    scene.reset()

    # small warm-up step to let assets settle
    for _ in range(5):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

    print("[INFO] Scene demo running.")
    print("Expected prims (example env_0):")
    print("  /World/envs/env_0/BRAVER_BIPED")
    print("  /World/envs/env_0/BRAVER_QUAD")

    # 6) Simulation loop
    dt = sim_cfg.dt
    while simulation_app.is_running():
        t0 = time.time()

        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        if args_cli.real_time:
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    simulation_app.close()


if __name__ == "__main__":
    main()
