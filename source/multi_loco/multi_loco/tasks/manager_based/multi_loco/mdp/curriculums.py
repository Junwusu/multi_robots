import torch
from collections.abc import Sequence
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.managers import RewardManager


def _active_ids(env):
    t = env.env_type  # 0=biped, 1=quad（按你工程约定）
    biped_ids = torch.nonzero(t == 0, as_tuple=False).squeeze(-1)
    quad_ids  = torch.nonzero(t == 1, as_tuple=False).squeeze(-1)
    return biped_ids, quad_ids


def terrain_levels_vel_type_weighted(
    env,
    env_ids: Sequence[int],
    # 两种机器人资产名
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    # 可选：对两种类型设置不同的“上难度距离阈值系数/下难度系数”
    # 默认保持与你原函数一致
    up_dist_frac_biped: float = 0.5,
    up_dist_frac_quad:  float = 0.5,
    down_req_frac_biped: float = 0.5,
    down_req_frac_quad:  float = 0.5,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Curriculum update for generator terrains, using the active robot (biped/quad) per env.

    It updates terrain origins/levels for `env_ids` and returns mean terrain level.
    """
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)

    # env_ids 转 tensor，便于做 mask/索引
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # 只在给定 env_ids 里做 type 分组
    biped_all, quad_all = _active_ids(env)
    # mask on env_ids
    is_biped = (env.env_type[env_ids_t] == 0)
    is_quad  = (env.env_type[env_ids_t] == 1)
    biped_ids = env_ids_t[is_biped]
    quad_ids  = env_ids_t[is_quad]

    move_up   = torch.zeros(env_ids_t.shape[0], device=env.device, dtype=torch.bool)
    move_down = torch.zeros_like(move_up)

    # --- biped 分支 ---
    if biped_ids.numel() > 0:
        asset: Articulation = env.scene[biped_cfg.name]
        distance = torch.norm(
            asset.data.root_pos_w[biped_ids, :2] - env.scene.env_origins[biped_ids, :2],
            dim=1
        )
        up_th = terrain.cfg.terrain_generator.size[0] * up_dist_frac_biped
        mu = distance > up_th

        req = torch.norm(command[biped_ids, :2], dim=1) * env.max_episode_length_s * down_req_frac_biped
        md = distance < req
        md = md & (~mu)

        # 写回到 env_ids_t 对应位置
        move_up[is_biped] = mu
        move_down[is_biped] = md

    # --- quad 分支 ---
    if quad_ids.numel() > 0:
        asset: Articulation = env.scene[quad_cfg.name]
        distance = torch.norm(
            asset.data.root_pos_w[quad_ids, :2] - env.scene.env_origins[quad_ids, :2],
            dim=1
        )
        up_th = terrain.cfg.terrain_generator.size[0] * up_dist_frac_quad
        mu = distance > up_th

        req = torch.norm(command[quad_ids, :2], dim=1) * env.max_episode_length_s * down_req_frac_quad
        md = distance < req
        md = md & (~mu)

        move_up[is_quad] = mu
        move_down[is_quad] = md

    # 用原 API 更新：注意这里传的是 env_ids（全体），以及对应的 move_up/down（同长度）
    terrain.update_env_origins(env_ids_t, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_vel_tracking_type_weighted(
    env,
    env_ids: Sequence[int],
    # 两种资产
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    # 允许 biped/quad 用不同阈值（你也可以都给同一个值）
    level_up_threshold_biped: float = 0.75,
    level_down_threshold_biped: float = 0.5,
    level_up_threshold_quad: float = 0.75,
    level_down_threshold_quad: float = 0.5,
    # 你用到的 reward term 名（保持可配置，避免你改名字后这里崩）
    lin_term_name: str = "track_lin_vel_xy_exp",
    ang_term_name: str = "track_ang_vel_z_exp",
) -> torch.Tensor:
    """
    Curriculum update based on:
      - walked distance (computed from active robot asset per env)
      - velocity tracking rewards averaged over episode time (from reward_manager episode_sums)

    Works with generator terrain only.
    """
    terrain: TerrainImporter = env.scene.terrain
    reward: RewardManager = env.reward_manager

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # --- 从 reward_manager 拿 tracking 的 episode 平均值（按 env_ids）---
    # 注意：episode_sums 通常是每个 env 累积的标量；这里不需要区分 biped/quad
    lin_sum = reward._episode_sums[lin_term_name][env_ids_t] / env.max_episode_length_s
    ang_sum = reward._episode_sums[ang_term_name][env_ids_t] / env.max_episode_length_s

    lin_idx = reward._term_names.index(lin_term_name)
    ang_idx = reward._term_names.index(ang_term_name)
    lin_w = reward._term_cfgs[lin_idx].weight
    ang_w = reward._term_cfgs[ang_idx].weight

    # --- 在 env_ids 内部分 biped/quad ---
    is_biped = (env.env_type[env_ids_t] == 0)
    is_quad  = (env.env_type[env_ids_t] == 1)

    move_up   = torch.zeros(env_ids_t.shape[0], device=env.device, dtype=torch.bool)
    move_down = torch.zeros_like(move_up)

    # --- biped 分支：用 biped 资产算 distance ---
    if is_biped.any():
        b_ids = env_ids_t[is_biped]
        asset: Articulation = env.scene[biped_cfg.name]
        dist = torch.norm(asset.data.root_pos_w[b_ids, :2] - env.scene.env_origins[b_ids, :2], dim=1)

        mu = (
            (dist > terrain.cfg.terrain_generator.size[0] / 2)
            & (lin_sum[is_biped] > lin_w * level_up_threshold_biped)
            & (ang_sum[is_biped] > ang_w * level_up_threshold_biped)
        )

        md = (
            (lin_sum[is_biped] < lin_w * level_down_threshold_biped)
            | (ang_sum[is_biped] < ang_w * level_down_threshold_biped)
        )
        md = md & (~mu)

        move_up[is_biped] = mu
        move_down[is_biped] = md

    # --- quad 分支：用 quad 资产算 distance ---
    if is_quad.any():
        q_ids = env_ids_t[is_quad]
        asset: Articulation = env.scene[quad_cfg.name]
        dist = torch.norm(asset.data.root_pos_w[q_ids, :2] - env.scene.env_origins[q_ids, :2], dim=1)

        mu = (
            (dist > terrain.cfg.terrain_generator.size[0] / 2)
            & (lin_sum[is_quad] > lin_w * level_up_threshold_quad)
            & (ang_sum[is_quad] > ang_w * level_up_threshold_quad)
        )

        md = (
            (lin_sum[is_quad] < lin_w * level_down_threshold_quad)
            | (ang_sum[is_quad] < ang_w * level_down_threshold_quad)
        )
        md = md & (~mu)

        move_up[is_quad] = mu
        move_down[is_quad] = md

    # 更新课程（一次调用）
    terrain.update_env_origins(env_ids_t, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())






