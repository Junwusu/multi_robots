from __future__ import annotations
import math
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils
from typing import Literal
from isaaclab.managers import ManagerTermBase, EventTermCfg
from isaaclab.assets import RigidObject, Articulation

def sample_env_type(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ratio_quad: float = 0.33,
    ratio_hex: float = 0.33,
) -> None:
    """Randomly assign env_type per env on reset.
    env.env_type: LongTensor [num_envs], 0=biped, 1=quad, 2=hexapod
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

    # fixed-count assignment: exactly k envs are quad/hex
    k_quad = int(round(num * ratio_quad))
    k_hex = int(round(num * ratio_hex))
    perm = torch.randperm(num, device=device)

    env.env_type[env_ids] = 0  # all biped first
    env.env_type[env_ids[perm[:k_quad]]] = 1  # pick k as quad
    if k_hex > 0:
        hex_start = k_quad
        hex_end = min(k_quad + k_hex, num)
        env.env_type[env_ids[perm[hex_start:hex_end]]] = 2  # pick k as hexapod

    if not hasattr(env, "_dbg_env_type_once"):
        env._dbg_env_type_once = True
        u, c = torch.unique(env.env_type, return_counts=True)
        print(
            "[DEBUG sample_env_type] unique:",
            dict(zip(u.tolist(), c.tolist())),
            "ratio_quad=",
            ratio_quad,
            "ratio_hex=",
            ratio_hex,
        )



def hide_inactive_robot(
    env,
    env_ids: torch.Tensor,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
    hex_cfg: SceneEntityCfg,
    hidden_pos1=(1.0, 0.0, -100.0),
    hidden_pos2=(-1.0, 0.0, -100.0),
) -> None:
    """Move inactive robot far away (underground) to avoid contact interference."""
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type first.")

    device = env.device
    hidden_pos_t1 = torch.tensor(hidden_pos1, device=device).view(1, 3)
    hidden_pos_t2 = torch.tensor(hidden_pos2, device=device).view(1, 3)

    biped: Articulation = env.scene[biped_cfg.name]
    quad: Articulation = env.scene[quad_cfg.name]
    hexa: Articulation = env.scene[hex_cfg.name]

    # which envs are biped/quad/hex active?
    is_quad = env.env_type[env_ids] == 1
    is_hex = env.env_type[env_ids] == 2
    quad_ids = env_ids[is_quad]
    hex_ids = env_ids[is_hex]
    biped_ids = env_ids[~(is_quad | is_hex)]

    # inactive in quad envs: biped/hex should be hidden
    if quad_ids.numel() > 0:
        _set_root_to_hidden(biped, quad_ids, hidden_pos_t1)
        _set_root_to_hidden(hexa, quad_ids, hidden_pos_t2)

    # inactive in biped envs: quad/hex should be hidden
    if biped_ids.numel() > 0:
        _set_root_to_hidden(quad, biped_ids, hidden_pos_t1)
        _set_root_to_hidden(hexa, biped_ids, hidden_pos_t2)

    # inactive in hex envs: biped/quad should be hidden
    if hex_ids.numel() > 0:
        _set_root_to_hidden(biped, hex_ids, hidden_pos_t1)
        _set_root_to_hidden(quad, hex_ids, hidden_pos_t2)


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
    hex_cfg: SceneEntityCfg,
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
    hexa: Articulation = env.scene[hex_cfg.name]

    # resolve joint ids in the order you pass in
    biped_joint_ids = biped.find_joints(biped_cfg.joint_names, preserve_order=True)[0]
    quad_joint_ids = quad.find_joints(quad_cfg.joint_names, preserve_order=True)[0]
    hex_joint_ids = hexa.find_joints(hex_cfg.joint_names, preserve_order=True)[0]

    # split envs
    is_quad = env.env_type[env_ids] == 1
    is_hex = env.env_type[env_ids] == 2
    quad_ids = env_ids[is_quad]
    hex_ids = env_ids[is_hex]
    biped_ids = env_ids[~(is_quad | is_hex)]

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
        elif which == "quad":
            if (not hasattr(env, "action_default_quad")) or (env.action_default_quad is None):
                env.action_default_quad = torch.zeros((env.num_envs, len(joint_ids)), device=device, dtype=torch.float32)
            env.action_default_quad[ids] = q
        else:  # "hex"
            if (not hasattr(env, "action_default_hex")) or (env.action_default_hex is None):
                env.action_default_hex = torch.zeros((env.num_envs, len(joint_ids)), device=device, dtype=torch.float32)
            env.action_default_hex[ids] = q

    _reset(biped, biped_joint_ids, biped_ids, "biped")
    _reset(quad,  quad_joint_ids,  quad_ids,  "quad")
    _reset(hexa,  hex_joint_ids,   hex_ids,   "hex")




def compute_act_mask(env: ManagerBasedEnv, env_ids: torch.Tensor, action_dim: int = 18):
    device = env.device
    if not hasattr(env, "act_mask") or env.act_mask is None:
        env.act_mask = torch.ones((env.num_envs, action_dim), device=device, dtype=torch.float32)

    # 默认全 0，再按类型填
    env.act_mask[env_ids] = 0.0

    # 0=biped, 1=quad, 2=hex
    biped_ids = env_ids[env.env_type[env_ids] == 0]
    quad_ids  = env_ids[env.env_type[env_ids] == 1]
    hex_ids  = env_ids[env.env_type[env_ids] == 2]

    env.act_mask[biped_ids, :6] = 1.0
    env.act_mask[quad_ids, :12] = 1.0
    env.act_mask[hex_ids, :18] = 1.0
    if not hasattr(env, "_dbg_mask_once"):
        env._dbg_mask_once = True
        m = env.act_mask
        et = env.env_type
        print(
            "[DEBUG mask] biped mask mean:",
            m[et == 0].float().mean().item(),
            "quad mask mean:",
            m[et == 1].float().mean().item(),
            "hex mask mean:",
            m[et == 2].float().mean().item(),
        )
        print("[DEBUG mask] first biped mask row:", m[et==0][0].int().tolist() if (et==0).any() else None)


def cache_action_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
    hex_cfg: SceneEntityCfg,
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
    hexa: Articulation = env.scene[hex_cfg.name]

    # ✅ MUST be resolved by manager: biped_cfg.joint_ids / quad_cfg.joint_ids
    if not hasattr(biped_cfg, "joint_ids") or biped_cfg.joint_ids is None:
        raise RuntimeError("biped_cfg.joint_ids is not resolved. Make sure SceneEntityCfg has joint_names and preserve_order=True.")
    if not hasattr(quad_cfg, "joint_ids") or quad_cfg.joint_ids is None:
        raise RuntimeError("quad_cfg.joint_ids is not resolved. Make sure SceneEntityCfg has joint_names and preserve_order=True.")
    if not hasattr(hex_cfg, "joint_ids") or hex_cfg.joint_ids is None:
        raise RuntimeError("hex_cfg.joint_ids is not resolved. Make sure SceneEntityCfg has joint_names and preserve_order=True.")

    biped_joint_ids = biped_cfg.joint_ids
    quad_joint_ids  = quad_cfg.joint_ids
    hex_joint_ids  = hex_cfg.joint_ids

    # lazy allocate
    if (not hasattr(env, "action_default_biped")) or (env.action_default_biped is None):
        env.action_default_biped = torch.zeros((env.num_envs, len(biped_joint_ids)), device=device, dtype=torch.float32)
    if (not hasattr(env, "action_default_quad")) or (env.action_default_quad is None):
        env.action_default_quad  = torch.zeros((env.num_envs, len(quad_joint_ids)),  device=device, dtype=torch.float32)
    if (not hasattr(env, "action_default_hex")) or (env.action_default_hex is None):
        env.action_default_hex  = torch.zeros((env.num_envs, len(hex_joint_ids)),  device=device, dtype=torch.float32)

    env.action_default_biped[env_ids] = biped.data.joint_pos[env_ids][:, biped_joint_ids]
    env.action_default_quad[env_ids]  = quad.data.joint_pos[env_ids][:, quad_joint_ids]
    env.action_default_hex[env_ids]  = hexa.data.joint_pos[env_ids][:, hex_joint_ids]

    
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
    hex_cfg: SceneEntityCfg,
):
    """Reset ONLY the active robot (biped/quad/hex) root state for the given env_ids."""
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type first.")

    is_quad = env.env_type[env_ids] == 1
    is_hex = env.env_type[env_ids] == 2
    quad_ids = env_ids[is_quad]
    hex_ids = env_ids[is_hex]
    biped_ids = env_ids[~(is_quad | is_hex)]

    if biped_ids.numel() > 0:
        _reset_root_state_uniform_one(env, biped_ids, pose_range, velocity_range, biped_cfg)

    if quad_ids.numel() > 0:
        _reset_root_state_uniform_one(env, quad_ids, pose_range, velocity_range, quad_cfg)

    if hex_ids.numel() > 0:
        _reset_root_state_uniform_one(env, hex_ids, pose_range, velocity_range, hex_cfg)


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
    hex_cfg: SceneEntityCfg,
):
    """Reset ONLY the active robot joints (biped/quad/hex) for the given env_ids."""
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type first.")

    is_quad = env.env_type[env_ids] == 1
    is_hex = env.env_type[env_ids] == 2
    quad_ids = env_ids[is_quad]
    hex_ids = env_ids[is_hex]
    biped_ids = env_ids[~(is_quad | is_hex)]

    if biped_ids.numel() > 0:
        _reset_joints_by_offset_set_one(env, biped_ids, position_range, velocity_range, biped_cfg)

    if quad_ids.numel() > 0:
        _reset_joints_by_offset_set_one(env, quad_ids, position_range, velocity_range, quad_cfg)

    if hex_ids.numel() > 0:
        _reset_joints_by_offset_set_one(env, hex_ids, position_range, velocity_range, hex_cfg)


def _active_ids_3(env, env_ids: torch.Tensor | None):
    """Return (biped_ids, quad_ids, hex_ids) as 1D Long tensors on env.device, filtered by env_ids if provided."""
    if env_ids is None:
        ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        ids = env_ids.to(device=env.device, dtype=torch.long)

    t = env.env_type[ids]
    b = ids[t == 0]
    q = ids[t == 1]
    h = ids[t == 2]
    return b, q, h

def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def randomize_rigid_body_mass_type_weighted(
    env,
    env_ids: torch.Tensor | None,
    # 三个机器人各自的 cfg（可以指定 body_names/body_ids）
    biped_cfg,
    quad_cfg,
    hex_cfg,
    # 三种机器人各自的随机化参数（你也可以传一样的）
    mass_params_biped: tuple[float, float],
    mass_params_quad: tuple[float, float],
    mass_params_hex: tuple[float, float],
    operation_biped: Literal["add", "scale", "abs"] = "scale",
    operation_quad:  Literal["add", "scale", "abs"] = "scale",
    operation_hex:   Literal["add", "scale", "abs"] = "scale",
    distribution_biped: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_quad:  Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_hex:   Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia_biped: bool = True,
    recompute_inertia_quad:  bool = True,
    recompute_inertia_hex:   bool = True,
):
    """
    Randomize rigid-body mass for biped/quad/hex assets in a single call, gated by env.env_type.
    Uses CPU tensors to assign masses/inertias (same caveat as original).
    """

    # --- helper: do one asset exactly like your original function does ---
    def _apply_one(asset_cfg, ids, mass_params, operation, distribution, recompute_inertia):
        if ids.numel() == 0:
            return

        # asset: RigidObject | Articulation
        asset = env.scene[asset_cfg.name]

        # NOTE: original uses CPU tensors for assignment
        ids_cpu = ids.detach().cpu()

        # resolve body indices
        if asset_cfg.body_ids == slice(None):
            body_ids_cpu = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids_cpu = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # get masses (num_envs, num_bodies) or (num_assets, num_bodies) depending on view; PhysX view handles it
        masses = asset.root_physx_view.get_masses()

        # reset to defaults for selected env/body (important!)
        masses[ids_cpu[:, None], body_ids_cpu] = asset.data.default_mass[ids_cpu[:, None], body_ids_cpu].clone()

        # randomize masses in-place via the same helper you原来用的
        # 你工程里应该有这个函数；如果在同文件就直接可见
        masses = _randomize_prop_by_op(
            masses,
            mass_params,
            ids_cpu,
            body_ids_cpu,
            operation=operation,
            distribution=distribution,
        )

        # set masses
        asset.root_physx_view.set_masses(masses, ids_cpu)

        # recompute inertia if needed (scale default inertia by mass ratio)
        if recompute_inertia:
            ratios = masses[ids_cpu[:, None], body_ids_cpu] / asset.data.default_mass[ids_cpu[:, None], body_ids_cpu]
            inertias = asset.root_physx_view.get_inertias()

            # articulation: (num_envs, num_bodies, 9); rigid object: (num_envs, 9)
            from isaaclab.assets import Articulation  # 避免类型名未导入时出错
            if isinstance(asset, Articulation):
                inertias[ids_cpu[:, None], body_ids_cpu] = asset.data.default_inertia[ids_cpu[:, None], body_ids_cpu] * ratios[..., None]
            else:
                # 对 rigid object：asset_cfg.body_ids 通常是 slice(None) 或单体；这里保持与你原实现一致
                inertias[ids_cpu] = asset.data.default_inertia[ids_cpu] * ratios
            asset.root_physx_view.set_inertias(inertias, ids_cpu)

    # --- split env_ids by env_type (0/1/2) ---
    biped_ids, quad_ids, hex_ids = _active_ids_3(env, env_ids)

    _apply_one(biped_cfg, biped_ids, mass_params_biped, operation_biped, distribution_biped, recompute_inertia_biped)
    _apply_one(quad_cfg,  quad_ids,  mass_params_quad,  operation_quad,  distribution_quad,  recompute_inertia_quad)
    _apply_one(hex_cfg,   hex_ids,   mass_params_hex,   operation_hex,   distribution_hex,   recompute_inertia_hex)

def randomize_rigid_body_mass_inertia_type_weighted(
    env,
    env_ids: torch.Tensor | None,
    # 三个资产 cfg
    biped_cfg,
    quad_cfg,
    hex_cfg,
    # 三种机器人各自随机参数（你也可以传一样的）
    params_biped: tuple[float, float],
    params_quad: tuple[float, float],
    params_hex: tuple[float, float],
    operation_biped: Literal["add", "scale", "abs"],
    operation_quad:  Literal["add", "scale", "abs"],
    operation_hex:   Literal["add", "scale", "abs"],
    distribution_biped: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_quad:  Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_hex:   Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize masses and scale inertias proportionally (mass-inertia coupled randomization),
    for biped/quad/hex assets in a single call, gated by env.env_type.

    Notes:
      - Uses CPU tensors for PhysX setter (same as original).
      - Inertia scaling here follows your original logic:
          new_masses = randomize(old_masses)
          scale = new_masses / old_masses
          inertias *= scale
    """

    def _apply_one(asset_cfg, ids, params, operation, distribution):
        if ids.numel() == 0:
            return

        asset = env.scene[asset_cfg.name]

        # CPU ids
        ids_cpu = ids.detach().cpu()

        # body ids on CPU
        if asset_cfg.body_ids == slice(None):
            body_ids_cpu = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids_cpu = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # read current masses & inertias (clone like original)
        inertias = asset.root_physx_view.get_inertias().clone()
        masses_old = asset.root_physx_view.get_masses().clone()

        # IMPORTANT: keep old masses for scale denominator
        masses_new = masses_old.clone()

        # randomize masses_new in-place (only for selected env/body, via helper)
        masses_new = _randomize_prop_by_op(
            masses_new,
            params,
            ids_cpu,
            body_ids_cpu,
            operation=operation,
            distribution=distribution,
        )

        # scale = new/old, and scale inertias (shape-handling same as your original)
        scale = masses_new / masses_old
        inertias = inertias * scale.unsqueeze(-1)

        # set back only for env_ids (PhysX view setter will apply subset)
        asset.root_physx_view.set_masses(masses_new, ids_cpu)
        asset.root_physx_view.set_inertias(inertias, ids_cpu)

    biped_ids, quad_ids, hex_ids = _active_ids_3(env, env_ids)

    _apply_one(biped_cfg, biped_ids, params_biped, operation_biped, distribution_biped)
    _apply_one(quad_cfg,  quad_ids,  params_quad,  operation_quad,  distribution_quad)
    _apply_one(hex_cfg,   hex_ids,   params_hex,   operation_hex,   distribution_hex)


def _split_env_ids_by_type(env, env_ids: torch.Tensor | None):
    """Return (biped_ids_cpu, quad_ids_cpu, hex_ids_cpu) on CPU (since PhysX setters use CPU tensors here)."""
    if env_ids is None:
        ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        ids = env_ids.to(device=env.device, dtype=torch.long)

    t = env.env_type[ids]
    b = ids[t == 0]
    q = ids[t == 1]
    h = ids[t == 2]
    return b, q, h

class randomize_rigid_body_material_type_weighted(ManagerTermBase):
    """
    三机器人版本：分别为 biped/quad/hex 采样 material buckets，然后在对应 env_ids 上随机分配。
    注意：PhysX material 总数限制依旧存在，bucket 数别太夸张。
    """

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)

        # 取 cfg 里三个资产 cfg
        self.biped_cfg: SceneEntityCfg = cfg.params["biped_cfg"]
        self.quad_cfg: SceneEntityCfg = cfg.params["quad_cfg"]
        self.hex_cfg: SceneEntityCfg = cfg.params["hex_cfg"]

        self.biped: RigidObject | Articulation = env.scene[self.biped_cfg.name]
        self.quad:  RigidObject | Articulation = env.scene[self.quad_cfg.name]
        self.hex:   RigidObject | Articulation = env.scene[self.hex_cfg.name]

        for a, name in [(self.biped, self.biped_cfg.name), (self.quad, self.quad_cfg.name), (self.hex, self.hex_cfg.name)]:
            if not isinstance(a, (RigidObject, Articulation)):
                raise ValueError(f"randomize_rigid_body_material_type_weighted not supported for asset '{name}' type={type(a)}")

        # --- 为三个机器人分别 sample buckets（只在 init 做一次）---
        def _make_buckets(prefix: str):
            static_range = cfg.params.get(f"static_friction_range_{prefix}", (1.0, 1.0))
            dynamic_range = cfg.params.get(f"dynamic_friction_range_{prefix}", (1.0, 1.0))
            restit_range = cfg.params.get(f"restitution_range_{prefix}", (0.0, 0.0))
            num_buckets = int(cfg.params.get(f"num_buckets_{prefix}", 1))
            make_consistent = bool(cfg.params.get(f"make_consistent_{prefix}", False))

            ranges = torch.tensor([static_range, dynamic_range, restit_range], device="cpu", dtype=torch.float32)
            buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
            if make_consistent:
                buckets[:, 1] = torch.min(buckets[:, 0], buckets[:, 1])
            return buckets, num_buckets

        self.buckets_biped, self.nb_biped = _make_buckets("biped")
        self.buckets_quad,  self.nb_quad  = _make_buckets("quad")
        self.buckets_hex,   self.nb_hex   = _make_buckets("hex")

        # 这里我直接用“整 asset 随机分配”的通用逻辑（你 cfg.body_names='.*'，所以 body_ids=all）
        # 如果你以后要对 Articulation + 部分 body 做特殊 shape 索引，再加回你原来的 num_shapes_per_body 解析逻辑。

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        # 下面这些参数必须“接得住”，即使不用，也要在签名里出现（否则会 unexpected keyword）
        biped_cfg: SceneEntityCfg,
        quad_cfg: SceneEntityCfg,
        hex_cfg: SceneEntityCfg,
        static_friction_range_biped: tuple[float, float],
        dynamic_friction_range_biped: tuple[float, float],
        restitution_range_biped: tuple[float, float],
        num_buckets_biped: int,
        make_consistent_biped: bool,

        static_friction_range_quad: tuple[float, float],
        dynamic_friction_range_quad: tuple[float, float],
        restitution_range_quad: tuple[float, float],
        num_buckets_quad: int,
        make_consistent_quad: bool,

        static_friction_range_hex: tuple[float, float],
        dynamic_friction_range_hex: tuple[float, float],
        restitution_range_hex: tuple[float, float],
        num_buckets_hex: int,
        make_consistent_hex: bool,
    ):
        b_ids, q_ids, h_ids = _split_env_ids_by_type(env, env_ids)

        def _apply(asset: RigidObject | Articulation, ids: torch.Tensor, buckets: torch.Tensor, nb: int):
            if ids.numel() == 0:
                return
            ids_cpu = ids.cpu()

            total_num_shapes = asset.root_physx_view.max_shapes
            bucket_ids = torch.randint(0, nb, (len(ids_cpu), total_num_shapes), device="cpu")
            material_samples = buckets[bucket_ids]  # [N, S, 3]

            materials = asset.root_physx_view.get_material_properties()
            materials[ids_cpu] = material_samples
            asset.root_physx_view.set_material_properties(materials, ids_cpu)

        _apply(self.biped, b_ids, self.buckets_biped, self.nb_biped)
        _apply(self.quad,  q_ids, self.buckets_quad,  self.nb_quad)
        _apply(self.hex,   h_ids, self.buckets_hex,   self.nb_hex)


def randomize_actuator_gains_type_weighted(
    env,
    env_ids: torch.Tensor | None,
    # 三个机器人各自 asset cfg（可指定 joint_names/joint_ids）
    biped_cfg,
    quad_cfg,
    hex_cfg,
    # 三种机器人各自 stiffness/damping 的分布参数（None 表示不改）
    stiffness_params_biped: tuple[float, float] | None = None,
    stiffness_params_quad:  tuple[float, float] | None = None,
    stiffness_params_hex:   tuple[float, float] | None = None,
    damping_params_biped: tuple[float, float] | None = None,
    damping_params_quad:  tuple[float, float] | None = None,
    damping_params_hex:   tuple[float, float] | None = None,
    # 三种机器人各自操作与分布（也可传一样）
    operation_biped: Literal["add", "scale", "abs"] = "abs",
    operation_quad:  Literal["add", "scale", "abs"] = "abs",
    operation_hex:   Literal["add", "scale", "abs"] = "abs",
    distribution_biped: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_quad:  Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_hex:   Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Type-weighted actuator gains randomization for biped/quad/hex.

    Behavior matches your original randomize_actuator_gains():
      - For each actuator, decide which joints to randomize based on asset_cfg.joint_ids and actuator.joint_indices
      - Reset those entries to default stiffness/damping before applying randomization
      - Write back to actuator tensors; for ImplicitActuator also write to sim
    """

    def _apply_one(
        ids: torch.Tensor,
        asset_cfg,
        stiffness_params,
        damping_params,
        operation,
        distribution,
    ):
        if ids.numel() == 0:
            return

        asset = env.scene[asset_cfg.name]  # Articulation
        if ids is None:
            ids_local = torch.arange(env.scene.num_envs, device=asset.device)
        else:
            ids_local = ids.to(device=asset.device, dtype=torch.long)

        def _randomize(data: torch.Tensor, params: tuple[float, float], actuator_indices):
            # 这里 dim_0_ids=None 表示对 batch 维全处理；我们只在 data[ids_local] 上操作，所以 OK
            return _randomize_prop_by_op(
                data, params, dim_0_ids=None, dim_1_ids=actuator_indices,
                operation=operation, distribution=distribution
            )

        for actuator in asset.actuators.values():
            # -------- 决定 actuator_indices（在该 actuator 内部的列索引）和 global_indices（映射到全关节索引）--------
            if isinstance(asset_cfg.joint_ids, slice):
                # asset_cfg 选择全部关节 -> 这个 actuator 内部全取
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                else:
                    global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
            elif isinstance(actuator.joint_indices, slice):
                # actuator 覆盖全部关节，但我们只随机 asset_cfg 指定的关节
                global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            else:
                # 两者都是 list -> 取交集
                actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
                asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if actuator_indices.numel() == 0:
                    continue
                global_indices = actuator_joint_indices[actuator_indices]

            # -------- stiffness --------
            if stiffness_params is not None:
                stiffness = actuator.stiffness[ids_local].clone()
                # reset to default for selected joints
                stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[ids_local][:, global_indices].clone()
                _randomize(stiffness, stiffness_params, actuator_indices)
                actuator.stiffness[ids_local] = stiffness

                # implicit actuator needs write to sim
                from isaaclab.actuators import ImplicitActuator
                if isinstance(actuator, ImplicitActuator):
                    asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=ids_local
                    )

            # -------- damping --------
            if damping_params is not None:
                damping = actuator.damping[ids_local].clone()
                damping[:, actuator_indices] = asset.data.default_joint_damping[ids_local][:, global_indices].clone()
                _randomize(damping, damping_params, actuator_indices)
                actuator.damping[ids_local] = damping

                from isaaclab.actuators import ImplicitActuator
                if isinstance(actuator, ImplicitActuator):
                    asset.write_joint_damping_to_sim(
                        damping, joint_ids=actuator.joint_indices, env_ids=ids_local
                    )

    b_ids, q_ids, h_ids = _split_env_ids_by_type(env, env_ids)

    _apply_one(b_ids, biped_cfg, stiffness_params_biped, damping_params_biped, operation_biped, distribution_biped)
    _apply_one(q_ids, quad_cfg,  stiffness_params_quad,  damping_params_quad,  operation_quad,  distribution_quad)
    _apply_one(h_ids, hex_cfg,   stiffness_params_hex,   damping_params_hex,   operation_hex,   distribution_hex)


def randomize_rigid_body_coms_type_weighted(
    env,
    env_ids: torch.Tensor | None,
    # 三个资产 cfg
    biped_cfg,
    quad_cfg,
    hex_cfg,
    # 三种机器人各自 COM 分布参数：((x_min,x_max),(y_min,y_max),(z_min,z_max))
    com_params_biped: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    com_params_quad:  tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    com_params_hex:   tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    # 三种机器人各自操作/分布（也可都一样）
    operation_biped: Literal["add", "scale", "abs"] = "add",
    operation_quad:  Literal["add", "scale", "abs"] = "add",
    operation_hex:   Literal["add", "scale", "abs"] = "add",
    distribution_biped: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_quad:  Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_hex:   Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize rigid body COM for biped/quad/hex assets, gated by env.env_type.
    Uses CPU tensors for PhysX setters (same as original).
    """

    def _apply_one(asset_cfg, ids_cpu, com_params, operation, distribution):
        if ids_cpu.numel() == 0:
            return

        asset = env.scene[asset_cfg.name]

        # resolve body indices on CPU
        if asset_cfg.body_ids == slice(None):
            body_ids_cpu = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids_cpu = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

        coms = asset.root_physx_view.get_coms().clone()  # shape: (num_envs, num_bodies, 3)

        # per-dim randomization (x,y,z)
        for dim in range(3):
            coms[..., dim] = _randomize_prop_by_op(
                coms[..., dim],
                com_params[dim],
                ids_cpu,
                body_ids_cpu,
                operation=operation,
                distribution=distribution,
            )

        asset.root_physx_view.set_coms(coms, ids_cpu)

    b_ids_cpu, q_ids_cpu, h_ids_cpu = _split_env_ids_by_type(env, env_ids)
    b_ids_cpu = b_ids_cpu.cpu()
    q_ids_cpu = q_ids_cpu.cpu() 
    h_ids_cpu = h_ids_cpu.cpu()
    _apply_one(biped_cfg, b_ids_cpu, com_params_biped, operation_biped, distribution_biped)
    _apply_one(quad_cfg,  q_ids_cpu, com_params_quad,  operation_quad,  distribution_quad)
    _apply_one(hex_cfg,   h_ids_cpu, com_params_hex,   operation_hex,   distribution_hex)


def apply_external_force_torque_stochastic_type_weighted(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    # 三个资产 cfg（可以各自指定 body_names/body_ids）
    biped_cfg: SceneEntityCfg,
    quad_cfg:  SceneEntityCfg,
    hex_cfg:   SceneEntityCfg,
    # 三种机器人各自的 range / probability
    force_range_biped: dict[str, tuple[float, float]],
    torque_range_biped: dict[str, tuple[float, float]],
    probability_biped: float,

    force_range_quad: dict[str, tuple[float, float]],
    torque_range_quad: dict[str, tuple[float, float]],
    probability_quad: float,

    force_range_hex: dict[str, tuple[float, float]],
    torque_range_hex: dict[str, tuple[float, float]],
    probability_hex: float,

    # 可选：每类型整体缩放（不想用就留 1.0）
    force_scale_biped: float = 1.0,
    torque_scale_biped: float = 1.0,
    force_scale_quad: float = 1.0,
    torque_scale_quad: float = 1.0,
    force_scale_hex: float = 1.0,
    torque_scale_hex: float = 1.0,
):
    """
    Apply stochastic external forces/torques to active robot type per env.
    Forces/torques are stored in each asset buffer and applied when asset.write_data_to_sim() is called.
    """

    b_ids, q_ids, h_ids = _split_env_ids_by_type(env, env_ids)

    def _apply_one(
        ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        force_range: dict[str, tuple[float, float]],
        torque_range: dict[str, tuple[float, float]],
        prob: float,
        f_scale: float,
        t_scale: float,
    ):
        if ids.numel() == 0:
            return

        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        # clear existing forces/torques for this asset (same as original)
        asset._external_force_b *= 0
        asset._external_torque_b *= 0

        # stochastic mask over envs of this type
        rnd = torch.rand(ids.shape, device=ids.device)
        masked_ids = ids[rnd < prob]
        if masked_ids.numel() == 0:
            return

        # number of bodies that will receive force/torque
        if isinstance(asset_cfg.body_ids, list):
            num_bodies = len(asset_cfg.body_ids)
        else:
            # slice(None) 或其它情况 -> 默认全 body
            num_bodies = asset.num_bodies

        size = (masked_ids.numel(), num_bodies, 3)

        fr_list = [force_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z"]]
        fr = torch.tensor(fr_list, device=asset.device, dtype=torch.float32)
        forces = math_utils.sample_uniform(fr[:, 0], fr[:, 1], size, asset.device) * f_scale

        tr_list = [torque_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z"]]
        tr = torch.tensor(tr_list, device=asset.device, dtype=torch.float32)
        torques = math_utils.sample_uniform(tr[:, 0], tr[:, 1], size, asset.device) * t_scale

        asset.set_external_force_and_torque(
            forces, torques, env_ids=masked_ids, body_ids=asset_cfg.body_ids
        )

    _apply_one(b_ids, biped_cfg, force_range_biped, torque_range_biped, probability_biped,
              force_scale_biped, torque_scale_biped)

    _apply_one(q_ids, quad_cfg, force_range_quad, torque_range_quad, probability_quad,
              force_scale_quad, torque_scale_quad)

    _apply_one(h_ids, hex_cfg, force_range_hex, torque_range_hex, probability_hex,
              force_scale_hex, torque_scale_hex)


