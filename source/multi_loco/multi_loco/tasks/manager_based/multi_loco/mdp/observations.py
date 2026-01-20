from __future__ import annotations
import torch

from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..multi_loco_env import MultiLocoEnv

def _active_ids(env: MultiLocoEnv):
    env_ids = torch.arange(env.num_envs, device=env.device)
    biped_ids = env_ids[env.env_type == 0]
    quad_ids  = env_ids[env.env_type == 1]
    return biped_ids, quad_ids

def _active_copy_3(env: MultiLocoEnv, biped_tensor, quad_tensor):
    """Helper: make (N,3) output by picking per-env from biped/quad tensors."""
    out = torch.zeros((env.num_envs, 3), device=env.device, dtype=biped_tensor.dtype)
    biped_ids, quad_ids = _active_ids(env)
    if biped_ids.numel() > 0:
        out[biped_ids] = biped_tensor[biped_ids]
    if quad_ids.numel() > 0:
        out[quad_ids] = quad_tensor[quad_ids]
    return out

def active_base_ang_vel(
    env: MultiLocoEnv,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg: SceneEntityCfg  = SceneEntityCfg("quad"),
) -> torch.Tensor:
    """(N,3) root angular velocity in base frame from the active robot."""
    biped = env.scene[biped_cfg.name]
    quad  = env.scene[quad_cfg.name]
    return _active_copy_3(env, biped.data.root_ang_vel_b, quad.data.root_ang_vel_b)

def active_projected_gravity(
    env: MultiLocoEnv,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg: SceneEntityCfg  = SceneEntityCfg("quad"),
) -> torch.Tensor:
    """(N,3) projected gravity in base frame from the active robot."""
    biped = env.scene[biped_cfg.name]
    quad  = env.scene[quad_cfg.name]
    return _active_copy_3(env, biped.data.projected_gravity_b, quad.data.projected_gravity_b)

def active_joint_pos_rel_12(
    env: MultiLocoEnv,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """(N,12) joint pos rel to default. biped fills first 6, quad fills first 12."""
    device = env.device
    out = torch.zeros((env.num_envs, 12), device=device)

    biped: Articulation = env.scene[biped_cfg.name]
    quad:  Articulation = env.scene[quad_cfg.name]

    bj = biped_cfg.joint_ids  # len 6
    qj = quad_cfg.joint_ids   # len 12

    biped_ids, quad_ids = _active_ids(env)

    if biped_ids.numel() > 0:
        rel = biped.data.joint_pos[biped_ids][:, bj] - biped.data.default_joint_pos[biped_ids][:, bj]
        out[biped_ids, :len(bj)] = rel

    if quad_ids.numel() > 0:
        rel = quad.data.joint_pos[quad_ids][:, qj] - quad.data.default_joint_pos[quad_ids][:, qj]
        out[quad_ids, :len(qj)] = rel

    return out

def active_joint_vel_12(
    env: MultiLocoEnv,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """(N,12) joint vel. biped fills first 6, quad fills first 12."""
    device = env.device
    out = torch.zeros((env.num_envs, 12), device=device)

    biped: Articulation = env.scene[biped_cfg.name]
    quad:  Articulation = env.scene[quad_cfg.name]

    bj = biped_cfg.joint_ids
    qj = quad_cfg.joint_ids

    biped_ids, quad_ids = _active_ids(env)

    if biped_ids.numel() > 0:
        out[biped_ids, :len(bj)] = biped.data.joint_vel[biped_ids][:, bj]
    if quad_ids.numel() > 0:
        out[quad_ids, :len(qj)] = quad.data.joint_vel[quad_ids][:, qj]

    return out

def last_action(env: MultiLocoEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions

def get_gait_phase(env: MultiLocoEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)

def get_gait_command(env: MultiLocoEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)

def generated_commands(env: MultiLocoEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def active_base_lin_vel(env, biped_cfg=SceneEntityCfg("biped"), quad_cfg=SceneEntityCfg("quad")):
    biped = env.scene[biped_cfg.name]
    quad  = env.scene[quad_cfg.name]
    return _active_copy_3(env, biped.data.root_lin_vel_b, quad.data.root_lin_vel_b)


def robot_type_onehot(env) -> torch.Tensor:
    """
    Returns robot type as one-hot:
    [1, 0] = biped
    [0, 1] = quad
    Shape: (num_envs, 2)
    """
    # env.env_type: (num_envs,) long, 0 or 1
    env_type = env.env_type

    onehot = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    onehot[env_type == 0, 0] = 1.0  # biped
    onehot[env_type == 1, 1] = 1.0  # quad
    return onehot


def action_mask_12(env) -> torch.Tensor:
    # env.act_mask: [num_envs, 12]
    return env.act_mask



