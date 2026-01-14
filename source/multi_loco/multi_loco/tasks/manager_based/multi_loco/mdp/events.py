from __future__ import annotations
import math
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils


def sample_env_type(env: ManagerBasedEnv, env_ids: torch.Tensor, ratio_quad: float = 0.5) -> None:
    """Randomly assign env_type per env on reset.
    env.env_type: LongTensor [num_envs], 0=biped, 1=quad
    """
    device = env.device
    num = env_ids.numel()

    # lazy init
    if not hasattr(env, "env_type") or env.env_type is None:
        env.env_type = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    # # sample
    # # quad with prob ratio_quad
    # rnd = torch.rand(num, device=device)
    # env.env_type[env_ids] = (rnd < ratio_quad).long()

    # fixed-count assignment: exactly k envs are quad
    k = int(round(num * ratio_quad))  # ratio_quad=0.5 -> exactly half (up to rounding)
    perm = torch.randperm(num, device=device)

    env.env_type[env_ids] = 0  # all biped first
    env.env_type[env_ids[perm[:k]]] = 1  # pick k as quad

    if not hasattr(env, "_dbg_env_type_once"):
        env._dbg_env_type_once = True
        u, c = torch.unique(env.env_type, return_counts=True)
        print("[DEBUG sample_env_type] unique:", dict(zip(u.tolist(), c.tolist())), "ratio_quad=", ratio_quad)



def hide_inactive_robot(
    env,
    env_ids: torch.Tensor,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
    hidden_pos=(0.0, 0.0, -100.0),
) -> None:
    """Move inactive robot far away (underground) to avoid contact interference."""
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type first.")

    device = env.device
    hidden_pos_t = torch.tensor(hidden_pos, device=device).view(1, 3)

    biped: Articulation = env.scene[biped_cfg.name]
    quad: Articulation = env.scene[quad_cfg.name]

    # which envs are biped-active or quad-active?
    is_quad = env.env_type[env_ids] == 1
    quad_ids = env_ids[is_quad]          # envs where quad is active
    biped_ids = env_ids[~is_quad]        # envs where biped is active

    # inactive in quad envs: biped should be hidden
    if quad_ids.numel() > 0:
        _set_root_to_hidden(biped, quad_ids, hidden_pos_t)

    # inactive in biped envs: quad should be hidden
    if biped_ids.numel() > 0:
        _set_root_to_hidden(quad, biped_ids, hidden_pos_t)


def _set_root_to_hidden(art: Articulation, env_ids: torch.Tensor, hidden_pos_t: torch.Tensor) -> None:
    """Helper: set root pose/vel for an articulation for given env_ids."""
    # Many IsaacLab versions store root states in art.data.
    # We'll build a target root pose: position = hidden_pos, rotation = identity.
    device = art.data.root_pos_w.device
    n = env_ids.numel()

    pos = hidden_pos_t.repeat(n, 1).to(device)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 4).repeat(n, 1)


    root_pose = torch.cat([pos, quat], dim=-1)   # (N, 7)

    # zero velocity to avoid flying back
    lin_vel = torch.zeros((n, 3), device=device)
    ang_vel = torch.zeros((n, 3), device=device)

    root_velocity = torch.cat([lin_vel, ang_vel], dim=-1)  # (N, 6)

    # --- Option 1 (many versions): set_root_pose / set_root_velocity ---
    if hasattr(art, "set_root_pose"):
        art.set_root_pose(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        if hasattr(art, "set_root_velocity"):
            art.set_root_velocity(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)
        return

    # --- Option 2 (common newer pattern): write_root_pose_to_sim / write_root_velocity_to_sim ---
    if hasattr(art, "write_root_pose_to_sim"):
        art.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        if hasattr(art, "write_root_velocity_to_sim"):
            art.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        return

    # --- Option 3: set_root_state (pos, quat, linvel, angvel) ---
    if hasattr(art, "set_root_state"):
        root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)  # (n, 13)
        art.set_root_state(root_state, env_ids=env_ids)
        return

    raise AttributeError(
        "Cannot find a supported API to set articulation root pose. "
        "Check your IsaacLab version for Articulation root setter methods."
    )

def reset_active_joints_by_offset(
    env,
    env_ids: torch.Tensor,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
    position_range=(-0.15, 0.15),
    velocity_range=(-0.10, 0.10),
) -> None:
    """Reset ONLY the active robot joints (depends on env.env_type).
    - biped: reset the joints in biped_cfg.joint_names
    - quad : reset the joints in quad_cfg.joint_names
    """
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type first.")

    device = env.device

    biped: Articulation = env.scene[biped_cfg.name]
    quad: Articulation = env.scene[quad_cfg.name]

    # resolve joint ids in the order you pass in
    biped_joint_ids = biped.find_joints(biped_cfg.joint_names, preserve_order=True)[0]
    quad_joint_ids = quad.find_joints(quad_cfg.joint_names, preserve_order=True)[0]

    # split envs
    is_quad = env.env_type[env_ids] == 1
    quad_ids = env_ids[is_quad]
    biped_ids = env_ids[~is_quad]

    def _reset(art: Articulation, joint_ids, ids: torch.Tensor, which: str):
        if ids.numel() == 0:
            return
        n = ids.numel()

        q0 = art.data.default_joint_pos[ids][:, joint_ids]
        qd0 = art.data.default_joint_vel[ids][:, joint_ids]

        dq = (position_range[0] + (position_range[1] - position_range[0]) * torch.rand((n, len(joint_ids)), device=device))
        dqd = (velocity_range[0] + (velocity_range[1] - velocity_range[0]) * torch.rand((n, len(joint_ids)), device=device))

        q = q0 + dq
        qd = qd0 + dqd

        # write to sim if available
        if hasattr(art, "write_joint_state_to_sim"):
            art.write_joint_state_to_sim(q, qd, joint_ids=joint_ids, env_ids=ids)
        elif hasattr(art, "set_joint_state"):
            art.set_joint_state(q, qd, joint_ids=joint_ids, env_ids=ids)
        else:
            # fallback: at least set targets
            art.set_joint_position_target(q, joint_ids=joint_ids, env_ids=ids)
        
            # ✅ cache baseline = the q we *intended* to set (most reliable)
        if which == "biped":
            if (not hasattr(env, "action_default_biped")) or (env.action_default_biped is None):
                env.action_default_biped = torch.zeros((env.num_envs, len(joint_ids)), device=device, dtype=torch.float32)
            env.action_default_biped[ids] = q
        else:  # "quad"
            if (not hasattr(env, "action_default_quad")) or (env.action_default_quad is None):
                env.action_default_quad = torch.zeros((env.num_envs, len(joint_ids)), device=device, dtype=torch.float32)
            env.action_default_quad[ids] = q

    _reset(biped, biped_joint_ids, biped_ids, "biped")
    _reset(quad,  quad_joint_ids,  quad_ids,  "quad")




def compute_act_mask(env: ManagerBasedEnv, env_ids: torch.Tensor, action_dim: int = 12):
    device = env.device
    if not hasattr(env, "act_mask") or env.act_mask is None:
        env.act_mask = torch.ones((env.num_envs, action_dim), device=device, dtype=torch.float32)

    # 默认全 0，再按类型填
    env.act_mask[env_ids] = 0.0

    # 0=biped, 1=quad
    biped_ids = env_ids[env.env_type[env_ids] == 0]
    quad_ids  = env_ids[env.env_type[env_ids] == 1]

    env.act_mask[biped_ids, :6] = 1.0
    env.act_mask[quad_ids, :12] = 1.0
    if not hasattr(env, "_dbg_mask_once"):
        env._dbg_mask_once = True
        m = env.act_mask
        et = env.env_type
        print("[DEBUG mask] biped mask mean:", m[et==0].float().mean().item(),
            "quad mask mean:", m[et==1].float().mean().item())
        print("[DEBUG mask] first biped mask row:", m[et==0][0].int().tolist() if (et==0).any() else None)


def cache_action_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
) -> None:
    """Cache per-env action baseline (joint positions) onto env.

    Creates:
      env.action_default_biped: (num_envs, 6)
      env.action_default_quad : (num_envs, 12)
    Only updates the env_ids passed in.
    """
    device = env.device

    biped: Articulation = env.scene[biped_cfg.name]
    quad: Articulation = env.scene[quad_cfg.name]

    # ✅ MUST be resolved by manager: biped_cfg.joint_ids / quad_cfg.joint_ids
    if not hasattr(biped_cfg, "joint_ids") or biped_cfg.joint_ids is None:
        raise RuntimeError("biped_cfg.joint_ids is not resolved. Make sure SceneEntityCfg has joint_names and preserve_order=True.")
    if not hasattr(quad_cfg, "joint_ids") or quad_cfg.joint_ids is None:
        raise RuntimeError("quad_cfg.joint_ids is not resolved. Make sure SceneEntityCfg has joint_names and preserve_order=True.")

    biped_joint_ids = biped_cfg.joint_ids
    quad_joint_ids  = quad_cfg.joint_ids

    # lazy allocate
    if (not hasattr(env, "action_default_biped")) or (env.action_default_biped is None):
        env.action_default_biped = torch.zeros((env.num_envs, len(biped_joint_ids)), device=device, dtype=torch.float32)
    if (not hasattr(env, "action_default_quad")) or (env.action_default_quad is None):
        env.action_default_quad  = torch.zeros((env.num_envs, len(quad_joint_ids)),  device=device, dtype=torch.float32)

    env.action_default_biped[env_ids] = biped.data.joint_pos[env_ids][:, biped_joint_ids]
    env.action_default_quad[env_ids]  = quad.data.joint_pos[env_ids][:, quad_joint_ids]

    
def reset_low_base_counter(env, env_ids: torch.Tensor) -> None:
    device = env.device
    if (not hasattr(env, "low_base_counter")) or (env.low_base_counter is None):
        env.low_base_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
    env.low_base_counter[env_ids] = 0




def _reset_root_state_uniform_one(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()  # (n, 13): pos(3) quat(4) linvel(3) angvel(3)

    # --- pose noise ---
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in pose_keys], device=asset.device)
    pose_noise = math_utils.sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_noise[:, 0:3]
    delta_quat = math_utils.quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], delta_quat)

    # --- velocity noise ---
    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in vel_keys], device=asset.device)
    vel_noise = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + vel_noise

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_root_state_uniform_multi(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
):
    """Reset ONLY the active robot (biped or quad) root state for the given env_ids."""
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type first.")

    device = env.device
    is_quad = env.env_type[env_ids] == 1
    quad_ids = env_ids[is_quad]
    biped_ids = env_ids[~is_quad]

    if biped_ids.numel() > 0:
        _reset_root_state_uniform_one(env, biped_ids, pose_range, velocity_range, biped_cfg)

    if quad_ids.numel() > 0:
        _reset_root_state_uniform_one(env, quad_ids, pose_range, velocity_range, quad_cfg)



def _reset_joints_by_offset_set_one(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]

    # joint_ids 处理：如果你传了 joint_names=".*_joint" 这种，cfg 会带 joint_ids
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp to limits
    pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(pos_limits[..., 0], pos_limits[..., 1])

    vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-vel_limits, vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_by_offset_set_multi(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
):
    """Reset ONLY the active robot joints (biped or quad) for the given env_ids."""
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type first.")

    is_quad = env.env_type[env_ids] == 1
    quad_ids = env_ids[is_quad]
    biped_ids = env_ids[~is_quad]

    if biped_ids.numel() > 0:
        _reset_joints_by_offset_set_one(env, biped_ids, position_range, velocity_range, biped_cfg)

    if quad_ids.numel() > 0:
        _reset_joints_by_offset_set_one(env, quad_ids, position_range, velocity_range, quad_cfg)



