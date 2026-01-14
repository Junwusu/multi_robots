from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor
import math
from isaaclab.utils.math import euler_xyz_from_quat

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..multi_loco_env import MultiLocoEnv


def _get_active_asset(
    env: MultiLocoEnv,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
) -> RigidObject:
    """Return the active asset object based on env.env_type (0=biped, 1=quad)."""
    # Both assets exist in the scene; we select values with masks later.
    # Here return BOTH assets for convenience? We'll fetch separately in each function.
    raise NotImplementedError




def get_active_root_height(env, biped_cfg: SceneEntityCfg, quad_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return root height z for active robot in each env. Shape: (num_envs,)."""
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Add sample_env_type reset event first.")

    biped: Articulation = env.scene[biped_cfg.name]
    quad: Articulation = env.scene[quad_cfg.name]

    z_biped = biped.data.root_pos_w[:, 2]
    z_quad = quad.data.root_pos_w[:, 2]

    is_quad = env.env_type == 1
    return torch.where(is_quad, z_quad, z_biped)

def is_fallen(
    env: MultiLocoEnv,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
    min_height_biped: float = 0.18,
    min_height_quad: float = 0.22,
    grace_time: float = 0.2,
) -> torch.Tensor:
    z = get_active_root_height(env, biped_cfg, quad_cfg)

    # per-env min height
    min_h = torch.full_like(z, min_height_biped)
    min_h = torch.where(env.env_type == 1, torch.full_like(z, min_height_quad), min_h)

    # time since last reset (seconds)
    t = env.episode_length_buf.to(torch.float32) * float(env.step_dt)

    # IMPORTANT: grace_time prevents instant terminate
    return (z < min_h) & (t > grace_time)

    if not hasattr(env, "_dbg_fallen_once"):
        env._dbg_fallen_once = True
        print("[DBG fallen] z min/max biped:",
            biped.data.root_pos_w[:,2].min().item(), biped.data.root_pos_w[:,2].max().item())
        print("[DBG fallen] z min/max quad:",
            quad.data.root_pos_w[:,2].min().item(), quad.data.root_pos_w[:,2].max().item())


def illegal_contact_multi(
    env: MultiLocoEnv,
    biped_sensor_cfg: SceneEntityCfg,
    quad_sensor_cfg: SceneEntityCfg,
    threshold_biped: float = 1.0,
    threshold_quad: float = 1.0,
) -> torch.Tensor:
    """Terminate when illegal contact force exceeds threshold on ACTIVE robot (biped/quad).

    Args:
        biped_sensor_cfg: SceneEntityCfg(name="biped_contact_forces", body_names=[...])
        quad_sensor_cfg : SceneEntityCfg(name="quad_contact_forces",  body_names=[...])
        threshold_biped/quad: force threshold (N)

    Returns:
        done (N,) bool tensor
    """
    device = env.device
    N = env.num_envs

    # 必须有 env_type
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Call sample_env_type on reset first.")

    is_quad = (env.env_type == 1)  # (N,) bool

    # sensors
    b_sensor: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
    q_sensor: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]

    b_ids = getattr(biped_sensor_cfg, "body_ids", None)
    q_ids = getattr(quad_sensor_cfg, "body_ids", None)

    def _any_exceed(sensor: ContactSensor, ids, thr: float) -> torch.Tensor:
        if ids is None or len(ids) == 0:
            return torch.zeros((N,), device=device, dtype=torch.bool)
        # (N, T, nBodies, 3)
        f = sensor.data.net_forces_w_history[:, :, ids]
        # -> (N, nBodies) max over history of force norm
        m = torch.max(torch.linalg.norm(f, dim=-1), dim=1)[0]
        # -> (N,) any body exceed
        return torch.any(m > float(thr), dim=1)

    done_b = _any_exceed(b_sensor, b_ids, threshold_biped)
    done_q = _any_exceed(q_sensor, q_ids, threshold_quad)

    return torch.where(is_quad, done_q, done_b)


def bad_body_posture_multi(
    env: MultiLocoEnv,
    biped_cfg: SceneEntityCfg,
    quad_cfg: SceneEntityCfg,
    roll_threshold: float = math.pi / 6,
    pitch_threshold: float = math.pi / 4,
) -> torch.Tensor:
    """
    Return (num_envs,) bool: True means bad posture / fallen.

    Works for mixed biped+quad envs, selected by env.env_type (0=biped, 1=quad).
    Also keeps a persistent low-height counter on env.
    """
    # device = env.device
    # N = env.num_envs

    # --- select active asset per env ---
    if not hasattr(env, "env_type"):
        raise RuntimeError("env.env_type not found. Add sample_env_type reset event first.")
    is_quad = (env.env_type == 1)  # (N,) bool

    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]

    # root pose (select per env)
    quat_w = torch.where(
        is_quad.unsqueeze(-1),
        quad.data.root_com_quat_w,
        biped.data.root_com_quat_w,
    )
    # base_z = torch.where(
    #     is_quad,
    #     quad.data.root_pos_w[:, 2],
    #     biped.data.root_pos_w[:, 2],
    # )

    roll, pitch, yaw = euler_xyz_from_quat(quat_w)
    # print("quat_w[0] =", quat_w[0].detach().cpu().numpy())
    # print("roll/pitch[0] =", roll[0].item(), pitch[0].item())

    # # per-env height threshold
    # h_thr = torch.where(
    #     is_quad,
    #     torch.tensor(height_threshold_quad, device=device),
    #     torch.tensor(height_threshold_biped, device=device),
    # )
    # low_base = base_z < h_thr  # (N,) bool

    # # --- persistent counter (IMPORTANT) ---
    # # create once, then reuse
    # if not hasattr(env, counter_name) or getattr(env, counter_name) is None:
    #     setattr(env, counter_name, torch.zeros(N, dtype=torch.int32, device=device))

    # counter = getattr(env, counter_name)
    # # ensure correct shape (e.g., if num_envs changed)
    # if counter.shape[0] != N:
    #     counter = torch.zeros(N, dtype=torch.int32, device=device)

    # counter = torch.where(low_base, counter + 1, torch.zeros_like(counter))
    # setattr(env, counter_name, counter)

    # bad_envs = (
    #     (roll.abs() > roll_threshold)
    #     | (pitch.abs() > pitch_threshold)
    #     # | (counter > int(low_height_steps))
    # )

    bad_envs = (
        (roll < -roll_threshold)
        | (roll > roll_threshold)
        | (pitch < -pitch_threshold)
        | (pitch > pitch_threshold)
    )
    return bad_envs

