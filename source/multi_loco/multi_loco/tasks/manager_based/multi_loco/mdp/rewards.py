from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase,SceneEntityCfg
from isaaclab.sensors import RayCaster,ContactSensor 
from isaaclab.managers import RewardTermCfg
if TYPE_CHECKING:
    from ..multi_loco_env import MultiLocoEnv

from multi_loco.assets.robots import * 

def _active_ids(env):
    env_ids = torch.arange(env.num_envs, device=env.device)
    biped_ids = env_ids[env.env_type == 0]
    quad_ids  = env_ids[env.env_type == 1]
    hex_ids  = env_ids[env.env_type == 2]
    return biped_ids, quad_ids, hex_ids


"""
General.
"""

#用
def stay_alive_type_weighted(env, w_biped: float = 1.0, w_quad: float = 1.0, w_hex: float = 1.0) -> torch.Tensor:
    out = torch.zeros(env.num_envs, device=env.device)
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    if biped_ids.numel() > 0:
        out[biped_ids] = w_biped
    if quad_ids.numel() > 0:
        out[quad_ids] = w_quad
    if hex_ids.numel() > 0:
        out[hex_ids] = w_hex
    return out



"""
Root track rewards.
"""

#用
def track_lin_vel_xy_exp_type_weighted(
    env,
    command_name: str,
    std_biped: float,
    std_quad: float,
    std_hex: float,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    cmd = env.command_manager.get_command(command_name)  # (N,3)

    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]

    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        err = torch.sum((cmd[biped_ids, :2] - biped.data.root_lin_vel_b[biped_ids, :2]) ** 2, dim=1)
        out[biped_ids] = w_biped * torch.exp(-err / (std_biped ** 2))

    if quad_ids.numel() > 0:
        err = torch.sum((cmd[quad_ids, :2] - quad.data.root_lin_vel_b[quad_ids, :2]) ** 2, dim=1)
        out[quad_ids] = w_quad * torch.exp(-err / (std_quad ** 2))

    if hex_ids.numel() > 0:
        err = torch.sum((cmd[hex_ids, :2] - hexa.data.root_lin_vel_b[hex_ids, :2]) ** 2, dim=1)
        out[hex_ids] = w_hex * torch.exp(-err / (std_hex ** 2))

    return out
#用
def track_ang_vel_z_exp_type_weighted(
    env,
    command_name: str,
    std_biped: float,
    std_quad: float,
    std_hex: float,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    cmd = env.command_manager.get_command(command_name)  # (N,3)

    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]

    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        err = (cmd[biped_ids, 2] - biped.data.root_ang_vel_b[biped_ids, 2]) ** 2
        out[biped_ids] = w_biped * torch.exp(-err / (std_biped ** 2))

    if quad_ids.numel() > 0:
        err = (cmd[quad_ids, 2] - quad.data.root_ang_vel_b[quad_ids, 2]) ** 2
        out[quad_ids] = w_quad * torch.exp(-err / (std_quad ** 2))

    if hex_ids.numel() > 0:
        err = (cmd[hex_ids, 2] - hexa.data.root_ang_vel_b[hex_ids, 2]) ** 2
        out[hex_ids] = w_hex * torch.exp(-err / (std_hex ** 2))

    return out


"""
Root penalties.
"""

#用
def lin_vel_z_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        out[biped_ids] = w_biped * (biped.data.root_lin_vel_b[biped_ids, 2] ** 2)
    if quad_ids.numel() > 0:
        out[quad_ids]  = w_quad  * (quad.data.root_lin_vel_b[quad_ids, 2] ** 2)
    if hex_ids.numel() > 0:
        out[hex_ids]  = w_hex  * (hexa.data.root_lin_vel_b[hex_ids, 2] ** 2)

    return out
#用
def ang_vel_xy_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        out[biped_ids] = w_biped * torch.sum(biped.data.root_ang_vel_b[biped_ids, :2] ** 2, dim=1)
    if quad_ids.numel() > 0:
        out[quad_ids]  = w_quad  * torch.sum(quad.data.root_ang_vel_b[quad_ids, :2] ** 2, dim=1)
    if hex_ids.numel() > 0:
        out[hex_ids]  = w_hex  * torch.sum(hexa.data.root_ang_vel_b[hex_ids, :2] ** 2, dim=1)

    return out
#用
def base_com_height_abs_type_weighted(
    env,
    target_height_biped: float,
    target_height_quad: float,
    target_height_hex: float,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
    # 可选：如果你以后想 rough terrain，用各自的 scanner
    biped_sensor_cfg: SceneEntityCfg | None = None,
    quad_sensor_cfg:  SceneEntityCfg | None = None,
    hex_sensor_cfg:  SceneEntityCfg | None = None,
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)

    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]

    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        tgt = target_height_biped
        if biped_sensor_cfg is not None:
            s: RayCaster = env.scene.sensors[biped_sensor_cfg.name]
            tgt = tgt + torch.mean(s.data.ray_hits_w[biped_ids, ..., 2], dim=1)
        out[biped_ids] = w_biped * torch.abs(biped.data.root_pos_w[biped_ids, 2] - tgt)

    if quad_ids.numel() > 0:
        tgt = target_height_quad
        if quad_sensor_cfg is not None:
            s: RayCaster = env.scene.sensors[quad_sensor_cfg.name]
            tgt = tgt + torch.mean(s.data.ray_hits_w[quad_ids, ..., 2], dim=1)
        out[quad_ids] = w_quad * torch.abs(quad.data.root_pos_w[quad_ids, 2] - tgt)

    if hex_ids.numel() > 0:
        tgt = target_height_hex
        if hex_sensor_cfg is not None:
            s: RayCaster = env.scene.sensors[hex_sensor_cfg.name]
            tgt = tgt + torch.mean(s.data.ray_hits_w[hex_ids, ..., 2], dim=1)
        out[hex_ids] = w_hex * torch.abs(hexa.data.root_pos_w[hex_ids, 2] - tgt)

    return out
#用
def flat_orientation_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        out[biped_ids] = w_biped * torch.sum(biped.data.projected_gravity_b[biped_ids, :2] ** 2, dim=1)
    if quad_ids.numel() > 0:
        out[quad_ids]  = w_quad  * torch.sum(quad.data.projected_gravity_b[quad_ids, :2] ** 2, dim=1)
    if hex_ids.numel() > 0:
        out[hex_ids]  = w_hex  * torch.sum(hexa.data.projected_gravity_b[hex_ids, :2] ** 2, dim=1)

    return out



"""
Action penalties.
"""

#用
def action_rate_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    use_mask: bool = True,
) -> torch.Tensor:
    # action shape: (N, 18)
    a  = env.action_manager.action
    pa = env.action_manager.prev_action

    if use_mask and hasattr(env, "act_mask") and env.act_mask is not None:
        da = (a - pa) * env.act_mask
    else:
        da = (a - pa)

    per_env = torch.sum(da * da, dim=1)  # (N,)

    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros_like(per_env)
    if biped_ids.numel() > 0:
        out[biped_ids] = w_biped * per_env[biped_ids]
    if quad_ids.numel() > 0:
        out[quad_ids]  = w_quad  * per_env[quad_ids]
    if hex_ids.numel() > 0:
        out[hex_ids]  = w_hex  * per_env[hex_ids]
    return out

class ActionSmoothnessPenalty_type(ManagerTermBase):
    """Second-difference action smoothness penalty with type-specific weights.

    penalty_i = w_type(i) * || (a_t - 2 a_{t-1} + a_{t-2}) ||^2
    Optionally mask inactive action dims (e.g., biped last 6 dims).
    """

    def __init__(
        self,
        cfg: RewardTermCfg,
        env,
        w_biped: float = 1.0,
        w_quad: float = -0.00,
        w_hex: float = -0.00,
        use_mask: bool = True,
        warmup_steps: int = 5,   # 0=不额外按episode_length_buf屏蔽；你也可以设 3/5
        acc_clip: float = 10.0,  # 例如 10.0；None=不clip
    ):
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.w_biped = float(w_biped)
        self.w_quad = float(w_quad)
        self.w_hex = float(w_hex)
        self.use_mask = bool(use_mask)
        self.warmup_steps = int(warmup_steps)
        self.acc_clip = acc_clip

        self.prev_prev_action = None
        self.prev_action = None

    def __call__(self, env: MultiLocoEnv) -> torch.Tensor:
        current_action = env.action_manager.action.clone()  # (N,12)

        # 可选：mask 无效 action 维度（biped 后6维置0）
        if self.use_mask and hasattr(env, "act_mask") and env.act_mask is not None:
            current_action = current_action * env.act_mask

        # warmup：避免前几步惩罚（照你写法）
        if self.prev_action is None or self.prev_prev_action is None:
            self.prev_prev_action = current_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        acc = current_action - 2.0 * self.prev_action + self.prev_prev_action  # (N,12)

        # 可选：clip acc 本身（照你注释那种）
        if self.acc_clip is not None:
            acc = torch.clamp(acc, -float(self.acc_clip), float(self.acc_clip))

        penalty = torch.sum(acc * acc, dim=1)  # (N,)

        # 可选：前几步 episode 不惩罚（你注释里的 startup mask）
        if self.warmup_steps > 0 and hasattr(env, "episode_length_buf"):
            penalty = penalty.clone()
            penalty[env.episode_length_buf < self.warmup_steps] = 0.0

        # 类型权重（biped/quad 不同）
        if hasattr(env, "env_type"):
            et = env.env_type
            w = torch.full_like(penalty, self.w_biped, dtype=penalty.dtype)
            w = torch.where(et == 1, torch.full_like(penalty, self.w_quad, dtype=penalty.dtype), w)
            w = torch.where(et == 2, torch.full_like(penalty, self.w_hex, dtype=penalty.dtype), w)
            penalty = penalty * w

        # 更新历史（照你写法）
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        return penalty


"""
Contact sensor.
"""

#用
def undesired_contacts_type_weighted(
    env,
    threshold: float,
    w_biped: float,
    w_quad: float,
    w_hex: float,
    biped_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "biped_contact_forces", body_names=BRAVER_biped_UNDESIRED_CONTACTS_NAMES
    ),
    quad_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "quad_contact_forces", body_names=BRAVER_QUAD_UNDESIRED_CONTACTS_NAMES
    ),
    hex_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "hexapod_contact_forces", body_names=BRAVER_HEXAPOD_UNDESIRED_CONTACTS_NAMES
    ),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        s: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        net = s.data.net_forces_w_history[biped_ids][:, :, biped_sensor_cfg.body_ids]  # (Nb,H,2,3)
        is_contact = torch.max(torch.norm(net, dim=-1), dim=1)[0] > threshold          # (Nb,2)
        out[biped_ids] = w_biped * torch.sum(is_contact, dim=1)

    if quad_ids.numel() > 0:
        s: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        net = s.data.net_forces_w_history[quad_ids][:, :, quad_sensor_cfg.body_ids]   # (Nq,H,4,3)
        is_contact = torch.max(torch.norm(net, dim=-1), dim=1)[0] > threshold         # (Nq,4)
        out[quad_ids] = w_quad * torch.sum(is_contact, dim=1)

    if hex_ids.numel() > 0:
        s: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        net = s.data.net_forces_w_history[hex_ids][:, :, hex_sensor_cfg.body_ids]   # (Nh,H,6,3)
        is_contact = torch.max(torch.norm(net, dim=-1), dim=1)[0] > threshold       # (Nh,6)
        out[hex_ids] = w_hex * torch.sum(is_contact, dim=1)

    return out

def contact_forces_type_weighted(
    env,
    threshold_biped: float,
    threshold_quad: float,
    threshold_hex: float,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_sensor_cfg: SceneEntityCfg = SceneEntityCfg("biped_contact_forces"),
    quad_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("quad_contact_forces"),
    hex_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod_contact_forces"),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        net = cs.data.net_forces_w_history[biped_ids]  # [Nb,H,B,3]
        peak = torch.max(torch.norm(net[:, :, biped_sensor_cfg.body_ids], dim=-1), dim=1)[0]
        vio = peak - threshold_biped
        out[biped_ids] = w_biped * torch.sum(vio.clamp(min=0.0), dim=1)

    if quad_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        net = cs.data.net_forces_w_history[quad_ids]
        peak = torch.max(torch.norm(net[:, :, quad_sensor_cfg.body_ids], dim=-1), dim=1)[0]
        vio = peak - threshold_quad
        out[quad_ids] = w_quad * torch.sum(vio.clamp(min=0.0), dim=1)

    if hex_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        net = cs.data.net_forces_w_history[hex_ids]
        peak = torch.max(torch.norm(net[:, :, hex_sensor_cfg.body_ids], dim=-1), dim=1)[0]
        vio = peak - threshold_hex
        out[hex_ids] = w_hex * torch.sum(vio.clamp(min=0.0), dim=1)

    return out


"""
Joint penalties.
"""

#用
def joint_pos_limits_type_weighted(
    env,
    w_biped: float,
    w_quad: float,
    w_hex: float,
    biped_cfg: SceneEntityCfg,
    quad_cfg:  SceneEntityCfg,
    hex_cfg:  SceneEntityCfg,
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    biped: Articulation = env.scene[biped_cfg.name]
    quad:  Articulation = env.scene[quad_cfg.name]
    hexa:  Articulation = env.scene[hex_cfg.name]
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        if (not hasattr(biped_cfg, "joint_ids")) or (biped_cfg.joint_ids is None):
            raise RuntimeError("biped_cfg.joint_ids not resolved. Provide joint_names+preserve_order=True in SceneEntityCfg.")
        j = biped_cfg.joint_ids
        pos = biped.data.joint_pos[biped_ids][:, j]
        low = biped.data.soft_joint_pos_limits[biped_ids][:, j, 0]
        high= biped.data.soft_joint_pos_limits[biped_ids][:, j, 1]
        ool = -(pos - low).clamp(max=0.0) + (pos - high).clamp(min=0.0)
        out[biped_ids] = w_biped * torch.sum(ool, dim=1)

    if quad_ids.numel() > 0:
        if (not hasattr(quad_cfg, "joint_ids")) or (quad_cfg.joint_ids is None):
            raise RuntimeError("quad_cfg.joint_ids not resolved. Provide joint_names+preserve_order=True in SceneEntityCfg.")
        j = quad_cfg.joint_ids
        pos = quad.data.joint_pos[quad_ids][:, j]
        low = quad.data.soft_joint_pos_limits[quad_ids][:, j, 0]
        high= quad.data.soft_joint_pos_limits[quad_ids][:, j, 1]
        ool = -(pos - low).clamp(max=0.0) + (pos - high).clamp(min=0.0)
        out[quad_ids] = w_quad * torch.sum(ool, dim=1)

    if hex_ids.numel() > 0:
        if (not hasattr(hex_cfg, "joint_ids")) or (hex_cfg.joint_ids is None):
            raise RuntimeError("hex_cfg.joint_ids not resolved. Provide joint_names+preserve_order=True in SceneEntityCfg.")
        j = hex_cfg.joint_ids
        pos = hexa.data.joint_pos[hex_ids][:, j]
        low = hexa.data.soft_joint_pos_limits[hex_ids][:, j, 0]
        high= hexa.data.soft_joint_pos_limits[hex_ids][:, j, 1]
        ool = -(pos - low).clamp(max=0.0) + (pos - high).clamp(min=0.0)
        out[hex_ids] = w_hex * torch.sum(ool, dim=1)

    return out
#用
def joint_deviation_l1_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        a: Articulation = env.scene[biped_cfg.name]
        angle = a.data.joint_pos[biped_ids][:, biped_cfg.joint_ids] - a.data.default_joint_pos[biped_ids][:, biped_cfg.joint_ids]
        out[biped_ids] = w_biped * torch.sum(torch.abs(angle), dim=1)

    if quad_ids.numel() > 0:
        a: Articulation = env.scene[quad_cfg.name]
        angle = a.data.joint_pos[quad_ids][:, quad_cfg.joint_ids] - a.data.default_joint_pos[quad_ids][:, quad_cfg.joint_ids]
        out[quad_ids] = w_quad * torch.sum(torch.abs(angle), dim=1)

    if hex_ids.numel() > 0:
        a: Articulation = env.scene[hex_cfg.name]
        angle = a.data.joint_pos[hex_ids][:, hex_cfg.joint_ids] - a.data.default_joint_pos[hex_ids][:, hex_cfg.joint_ids]
        out[hex_ids] = w_hex * torch.sum(torch.abs(angle), dim=1)

    return out

def stand_still_joint_deviation_l1_type_weighted(
    env,
    command_name: str,
    command_threshold: float = 0.06,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    still_mask = (torch.norm(cmd[:, :2], dim=1) < command_threshold).float()

    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        a: Articulation = env.scene[biped_cfg.name]
        angle = a.data.joint_pos[biped_ids][:, biped_cfg.joint_ids] - a.data.default_joint_pos[biped_ids][:, biped_cfg.joint_ids]
        out[biped_ids] = w_biped * torch.sum(torch.abs(angle), dim=1) * still_mask[biped_ids]

    if quad_ids.numel() > 0:
        a: Articulation = env.scene[quad_cfg.name]
        angle = a.data.joint_pos[quad_ids][:, quad_cfg.joint_ids] - a.data.default_joint_pos[quad_ids][:, quad_cfg.joint_ids]
        out[quad_ids] = w_quad * torch.sum(torch.abs(angle), dim=1) * still_mask[quad_ids]

    if hex_ids.numel() > 0:
        a: Articulation = env.scene[hex_cfg.name]
        angle = a.data.joint_pos[hex_ids][:, hex_cfg.joint_ids] - a.data.default_joint_pos[hex_ids][:, hex_cfg.joint_ids]
        out[hex_ids] = w_hex * torch.sum(torch.abs(angle), dim=1) * still_mask[hex_ids]

    return out

def joint_power_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    """
    Type-weighted version of joint_power_l2.

    For each env:
      penalty = sum(|tau * qd|) over the joints specified by that type's cfg.joint_ids
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if biped_ids.numel() > 0:
        a: Articulation = env.scene[biped_cfg.name]
        jp = a.data.applied_torque[biped_ids][:, biped_cfg.joint_ids] * a.data.joint_vel[biped_ids][:, biped_cfg.joint_ids]
        out[biped_ids] = w_biped * torch.sum(torch.abs(jp), dim=1)

    if quad_ids.numel() > 0:
        a: Articulation = env.scene[quad_cfg.name]
        jp = a.data.applied_torque[quad_ids][:, quad_cfg.joint_ids] * a.data.joint_vel[quad_ids][:, quad_cfg.joint_ids]
        out[quad_ids] = w_quad * torch.sum(torch.abs(jp), dim=1)

    if hex_ids.numel() > 0:
        a: Articulation = env.scene[hex_cfg.name]
        jp = a.data.applied_torque[hex_ids][:, hex_cfg.joint_ids] * a.data.joint_vel[hex_ids][:, hex_cfg.joint_ids]
        out[hex_ids] = w_hex * torch.sum(torch.abs(jp), dim=1)

    return out
#用
def joint_vel_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    """
    Type-weighted joint velocity L2 penalty.

    For each env:
      penalty = sum( qd^2 ) over joints specified by that type's cfg.joint_ids
    Output is >=0; typically use a NEGATIVE RewTerm.weight in cfg to penalize it.
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if biped_ids.numel() > 0:
        a: Articulation = env.scene[biped_cfg.name]
        qd = a.data.joint_vel[biped_ids][:, biped_cfg.joint_ids]
        out[biped_ids] = w_biped * torch.sum(torch.square(qd), dim=1)

    if quad_ids.numel() > 0:
        a: Articulation = env.scene[quad_cfg.name]
        qd = a.data.joint_vel[quad_ids][:, quad_cfg.joint_ids]
        out[quad_ids] = w_quad * torch.sum(torch.square(qd), dim=1)

    if hex_ids.numel() > 0:
        a: Articulation = env.scene[hex_cfg.name]
        qd = a.data.joint_vel[hex_ids][:, hex_cfg.joint_ids]
        out[hex_ids] = w_hex * torch.sum(torch.square(qd), dim=1)

    return out
#用
def joint_acc_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    """
    Type-weighted joint acceleration L2 penalty.

    For each env:
      penalty = sum( qdd^2 ) over joints specified by that type's cfg.joint_ids
    Output is >=0; typically use a NEGATIVE RewTerm.weight in cfg to penalize it.
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if biped_ids.numel() > 0:
        a: Articulation = env.scene[biped_cfg.name]
        qdd = a.data.joint_acc[biped_ids][:, biped_cfg.joint_ids]
        out[biped_ids] = w_biped * torch.sum(torch.square(qdd), dim=1)

    if quad_ids.numel() > 0:
        a: Articulation = env.scene[quad_cfg.name]
        qdd = a.data.joint_acc[quad_ids][:, quad_cfg.joint_ids]
        out[quad_ids] = w_quad * torch.sum(torch.square(qdd), dim=1)

    if hex_ids.numel() > 0:
        a: Articulation = env.scene[hex_cfg.name]
        qdd = a.data.joint_acc[hex_ids][:, hex_cfg.joint_ids]
        out[hex_ids] = w_hex * torch.sum(torch.square(qdd), dim=1)

    return out
#用
def joint_torques_l2_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    """
    Type-weighted joint torque L2 penalty.

    For each env:
      penalty = sum( tau^2 ) over joints specified by that type's cfg.joint_ids
    Output is >=0; typically use a NEGATIVE RewTerm.weight in cfg to penalize it.
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if biped_ids.numel() > 0:
        a: Articulation = env.scene[biped_cfg.name]
        tau = a.data.applied_torque[biped_ids][:, biped_cfg.joint_ids]
        out[biped_ids] = w_biped * torch.sum(torch.square(tau), dim=1)

    if quad_ids.numel() > 0:
        a: Articulation = env.scene[quad_cfg.name]
        tau = a.data.applied_torque[quad_ids][:, quad_cfg.joint_ids]
        out[quad_ids] = w_quad * torch.sum(torch.square(tau), dim=1)

    if hex_ids.numel() > 0:
        a: Articulation = env.scene[hex_cfg.name]
        tau = a.data.applied_torque[hex_ids][:, hex_cfg.joint_ids]
        out[hex_ids] = w_hex * torch.sum(torch.square(tau), dim=1)

    return out
#用
def energy_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
) -> torch.Tensor:
    """
    Type-weighted energy penalty:
        sum(|qvel| * |torque|) over joints specified by each type's cfg.joint_ids

    Output is >=0; typically set a NEGATIVE RewTerm.weight in cfg to penalize energy usage.
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if biped_ids.numel() > 0:
        a: Articulation = env.scene[biped_cfg.name]
        qvel = a.data.joint_vel[biped_ids][:, biped_cfg.joint_ids]
        qfrc = a.data.applied_torque[biped_ids][:, biped_cfg.joint_ids]
        out[biped_ids] = w_biped * torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=1)

    if quad_ids.numel() > 0:
        a: Articulation = env.scene[quad_cfg.name]
        qvel = a.data.joint_vel[quad_ids][:, quad_cfg.joint_ids]
        qfrc = a.data.applied_torque[quad_ids][:, quad_cfg.joint_ids]
        out[quad_ids] = w_quad * torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=1)

    if hex_ids.numel() > 0:
        a: Articulation = env.scene[hex_cfg.name]
        qvel = a.data.joint_vel[hex_ids][:, hex_cfg.joint_ids]
        qfrc = a.data.applied_torque[hex_ids][:, hex_cfg.joint_ids]
        out[hex_ids] = w_hex * torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=1)

    return out

def joint_position_penalty_type_weighted(
    env,
    # biped/quad 内部权重
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    # biped/quad 各自参数（允许不同）
    stand_still_scale_biped: float = 1.0,
    stand_still_scale_quad: float = 1.0,
    stand_still_scale_hex: float = 1.0,
    velocity_threshold_biped: float = 0.0,
    velocity_threshold_quad: float = 0.0,
    velocity_threshold_hex: float = 0.0,
    # 资产 cfg
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped"),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad"),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Type-weighted joint position penalty (error from default).

    Original behavior per env:
      cmd = ||base_velocity_cmd||
      body_vel = ||root_lin_vel_b_xy||
      err = ||(q - q_default)||    (L2 norm over DOFs)
      if cmd>0 OR body_vel>vel_th:  out = err
      else:                        out = stand_still_scale * err

    This version applies the same logic separately for biped/quad, allowing different thresholds/scales/weights.
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)  # same for all

    def _compute(ids: torch.Tensor, asset_cfg: SceneEntityCfg, vel_th: float, stand_scale: float) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[ids, :2], dim=1)
        # 只用该资产的 DOF（joint_pos 本身就是该资产的 DOF）；如果你想只对部分关节惩罚，可用 asset_cfg.joint_ids
        err = torch.linalg.norm(
            (asset.data.joint_pos[ids] - asset.data.default_joint_pos[ids]),
            dim=1
        )
        moving = torch.logical_or(cmd[ids] > 0.0, body_vel > vel_th)
        return torch.where(moving, err, stand_scale * err)

    if biped_ids.numel() > 0:
        out[biped_ids] = w_biped * _compute(biped_ids, biped_cfg, velocity_threshold_biped, stand_still_scale_biped)

    if quad_ids.numel() > 0:
        out[quad_ids] = w_quad * _compute(quad_ids, quad_cfg, velocity_threshold_quad, stand_still_scale_quad)

    if hex_ids.numel() > 0:
        out[hex_ids] = w_hex * _compute(hex_ids, hex_cfg, velocity_threshold_hex, stand_still_scale_hex)

    return out



"""
Foot penalties.
"""

#用
def feet_distance_type_weighted(
    env,
    w_biped: float,
    w_quad: float,
    w_hex: float,
    # biped range
    biped_min: float = 0.2,
    biped_max: float = 0.56,
    # quad range (左右间距)
    quad_min: float = 0.12,
    quad_max: float = 0.65,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped",body_names=BRAVER_biped_FOOT_NAMES),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad",body_names=BRAVER_QUAD_FOOT_NAMES),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod",body_names=BRAVER_HEXAPOD_FOOT_NAMES),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    biped: Articulation = env.scene[biped_cfg.name]
    quad:  Articulation = env.scene[quad_cfg.name]
    hexa:  Articulation = env.scene[hex_cfg.name]

    if biped_ids.numel() > 0:
        idx = biped.find_bodies(BRAVER_biped_FOOT_NAMES)[0]  # [2]
        pos = biped.data.body_link_pos_w[biped_ids][:, idx, :2]  # (Nb,2,2)
        d = torch.norm(pos[:, 0] - pos[:, 1], dim=-1)
        r = torch.clip(biped_min - d, 0, 1) + torch.clip(d - biped_max, 0, 1)
        out[biped_ids] = w_biped * r

    if quad_ids.numel() > 0:
        idx = quad.find_bodies(BRAVER_QUAD_FOOT_NAMES)[0]  # [4] order: FL FR RL RR
        pos = quad.data.body_link_pos_w[quad_ids][:, idx, :2]  # (Nq,4,2)
        d_front = torch.norm(pos[:, 0] - pos[:, 1], dim=-1)  # FL-FR
        d_rear  = torch.norm(pos[:, 2] - pos[:, 3], dim=-1)  # RL-RR
        r_front = torch.clip(quad_min - d_front, 0, 1) + torch.clip(d_front - quad_max, 0, 1)
        r_rear  = torch.clip(quad_min - d_rear,  0, 1) + torch.clip(d_rear  - quad_max, 0, 1)
        out[quad_ids] = w_quad * (r_front + r_rear)

    if hex_ids.numel() > 0:
        idx = hexa.find_bodies(BRAVER_HEXAPOD_FOOT_NAMES)[0]  # [6]
        pos = hexa.data.body_link_pos_w[hex_ids][:, idx, :2]  # (Nh,6,2)
        d_pairs = []
        for i in range(0, 6, 2):
            d_pairs.append(torch.norm(pos[:, i] - pos[:, i + 1], dim=-1))
        d_pairs = torch.stack(d_pairs, dim=1)
        r_pairs = torch.clip(quad_min - d_pairs, 0, 1) + torch.clip(d_pairs - quad_max, 0, 1)
        out[hex_ids] = w_hex * torch.sum(r_pairs, dim=1)

    return out
#用
def feet_regulation_set_type_weighted(
    env,
    w_biped: float,
    w_quad: float,
    w_hex: float,
    foot_radius_biped: float,
    foot_radius_quad: float,
    foot_radius_hex: float,
    base_height_target_biped: float,
    base_height_target_quad: float,
    base_height_target_hex: float,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped", body_names=BRAVER_biped_FOOT_NAMES),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad",  body_names=BRAVER_QUAD_FOOT_NAMES),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod",  body_names=BRAVER_HEXAPOD_FOOT_NAMES),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]

    if biped_ids.numel() > 0:
        feet_h = torch.clip(biped.data.body_pos_w[biped_ids][:, biped_cfg.body_ids, 2] - foot_radius_biped, 0, 1)
        feet_v_xy = biped.data.body_lin_vel_w[biped_ids][:, biped_cfg.body_ids, :2]
        scale = torch.exp(-feet_h / (0.05 * base_height_target_biped))
        pen = torch.sum(scale * torch.square(torch.norm(feet_v_xy, dim=-1)), dim=1)
        out[biped_ids] = w_biped * torch.clip(pen, -2, 2)

    if quad_ids.numel() > 0:
        feet_h = torch.clip(quad.data.body_pos_w[quad_ids][:, quad_cfg.body_ids, 2] - foot_radius_quad, 0, 1)
        feet_v_xy = quad.data.body_lin_vel_w[quad_ids][:, quad_cfg.body_ids, :2]
        scale = torch.exp(-feet_h / (0.05 * base_height_target_quad))
        pen = torch.sum(scale * torch.square(torch.norm(feet_v_xy, dim=-1)), dim=1)
        out[quad_ids] = w_quad * torch.clip(pen, -2, 2)

    if hex_ids.numel() > 0:
        feet_h = torch.clip(hexa.data.body_pos_w[hex_ids][:, hex_cfg.body_ids, 2] - foot_radius_hex, 0, 1)
        feet_v_xy = hexa.data.body_lin_vel_w[hex_ids][:, hex_cfg.body_ids, :2]
        scale = torch.exp(-feet_h / (0.05 * base_height_target_hex))
        pen = torch.sum(scale * torch.square(torch.norm(feet_v_xy, dim=-1)), dim=1)
        out[hex_ids] = w_hex * torch.clip(pen, -2, 2)

    return out
#用
def foot_landing_vel_type_weighted(
    env,
    w_biped: float,
    w_quad: float,
    w_hex: float,
    foot_radius_biped: float,
    foot_radius_quad: float,
    foot_radius_hex: float,
    about_landing_threshold_biped: float,
    about_landing_threshold_quad: float,
    about_landing_threshold_hex: float,
    # contact sensor（用 net_forces_w 的 z 分量判断接触）
    biped_sensor_cfg: SceneEntityCfg = SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
    quad_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
    hex_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod_contact_forces",  body_names=BRAVER_HEXAPOD_FOOT_NAMES),
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped", body_names=BRAVER_biped_FOOT_NAMES),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad",  body_names=BRAVER_QUAD_FOOT_NAMES),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod",  body_names=BRAVER_HEXAPOD_FOOT_NAMES),
    contact_force_z_threshold: float = 2.0,
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    biped = env.scene[biped_cfg.name]
    quad  = env.scene[quad_cfg.name]
    hexa  = env.scene[hex_cfg.name]

    if biped_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        z_vels = biped.data.body_lin_vel_w[biped_ids][:, biped_cfg.body_ids, 2]
        contacts = cs.data.net_forces_w[biped_ids][:, biped_sensor_cfg.body_ids, 2] > contact_force_z_threshold
        foot_h = torch.clip(biped.data.body_pos_w[biped_ids][:, biped_cfg.body_ids, 2] - foot_radius_biped, 0, 1)
        about_to_land = (foot_h < about_landing_threshold_biped) & (~contacts) & (z_vels < 0.0)
        landing_z = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        pen = torch.sum(landing_z * landing_z, dim=1)
        out[biped_ids] = w_biped * pen

    if quad_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        z_vels = quad.data.body_lin_vel_w[quad_ids][:, quad_cfg.body_ids, 2]
        contacts = cs.data.net_forces_w[quad_ids][:, quad_sensor_cfg.body_ids, 2] > contact_force_z_threshold
        foot_h = torch.clip(quad.data.body_pos_w[quad_ids][:, quad_cfg.body_ids, 2] - foot_radius_quad, 0, 1)
        about_to_land = (foot_h < about_landing_threshold_quad) & (~contacts) & (z_vels < 0.0)
        landing_z = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        pen = torch.sum(landing_z * landing_z, dim=1)
        out[quad_ids] = w_quad * pen

    if hex_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        z_vels = hexa.data.body_lin_vel_w[hex_ids][:, hex_cfg.body_ids, 2]
        contacts = cs.data.net_forces_w[hex_ids][:, hex_sensor_cfg.body_ids, 2] > contact_force_z_threshold
        foot_h = torch.clip(hexa.data.body_pos_w[hex_ids][:, hex_cfg.body_ids, 2] - foot_radius_hex, 0, 1)
        about_to_land = (foot_h < about_landing_threshold_hex) & (~contacts) & (z_vels < 0.0)
        landing_z = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        pen = torch.sum(landing_z * landing_z, dim=1)
        out[hex_ids] = w_hex * pen

    return out
#用
def feet_velocity_y_abs_sum_type_weighted(
    env,
    w_biped: float,
    w_quad: float,
    w_hex: float,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped", body_names=BRAVER_biped_FOOT_NAMES),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad",  body_names=BRAVER_QUAD_FOOT_NAMES),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod",  body_names=BRAVER_HEXAPOD_FOOT_NAMES),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]

    if biped_ids.numel() > 0:
        v = biped.data.body_lin_vel_w[biped_ids][:, biped_cfg.body_ids, 1]
        out[biped_ids] = w_biped * torch.sum(torch.abs(v), dim=1)

    if quad_ids.numel() > 0:
        v = quad.data.body_lin_vel_w[quad_ids][:, quad_cfg.body_ids, 1]
        out[quad_ids] = w_quad * torch.sum(torch.abs(v), dim=1)

    if hex_ids.numel() > 0:
        v = hexa.data.body_lin_vel_w[hex_ids][:, hex_cfg.body_ids, 1]
        out[hex_ids] = w_hex * torch.sum(torch.abs(v), dim=1)

    return out
#用
def foot_clearance_reward1_type_weighted(
    env,
    w_biped: float,
    w_quad: float,
    w_hex: float,
    target_height_biped: float,
    target_height_quad: float,
    target_height_hex: float,
    std_biped: float,
    std_quad: float,
    std_hex: float,
    tanh_mult_biped: float,
    tanh_mult_quad: float,
    tanh_mult_hex: float,
    biped_cfg: SceneEntityCfg = SceneEntityCfg("biped", body_names=BRAVER_biped_FOOT_NAMES),
    quad_cfg:  SceneEntityCfg = SceneEntityCfg("quad",  body_names=BRAVER_QUAD_FOOT_NAMES),
    hex_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod",  body_names=BRAVER_HEXAPOD_FOOT_NAMES),
) -> torch.Tensor:
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    biped: RigidObject = env.scene[biped_cfg.name]
    quad:  RigidObject = env.scene[quad_cfg.name]
    hexa:  RigidObject = env.scene[hex_cfg.name]

    if biped_ids.numel() > 0:
        z_err = (biped.data.body_pos_w[biped_ids][:, biped_cfg.body_ids, 2] - target_height_biped) ** 2
        vxy = torch.norm(biped.data.body_lin_vel_w[biped_ids][:, biped_cfg.body_ids, :2], dim=2)
        v_tanh = torch.tanh(tanh_mult_biped * vxy)
        val = torch.exp(-torch.sum(z_err * v_tanh, dim=1) / std_biped)
        out[biped_ids] = w_biped * val

    if quad_ids.numel() > 0:
        z_err = (quad.data.body_pos_w[quad_ids][:, quad_cfg.body_ids, 2] - target_height_quad) ** 2
        vxy = torch.norm(quad.data.body_lin_vel_w[quad_ids][:, quad_cfg.body_ids, :2], dim=2)
        v_tanh = torch.tanh(tanh_mult_quad * vxy)
        val = torch.exp(-torch.sum(z_err * v_tanh, dim=1) / std_quad)
        out[quad_ids] = w_quad * val

    if hex_ids.numel() > 0:
        z_err = (hexa.data.body_pos_w[hex_ids][:, hex_cfg.body_ids, 2] - target_height_hex) ** 2
        vxy = torch.norm(hexa.data.body_lin_vel_w[hex_ids][:, hex_cfg.body_ids, :2], dim=2)
        v_tanh = torch.tanh(tanh_mult_hex * vxy)
        val = torch.exp(-torch.sum(z_err * v_tanh, dim=1) / std_hex)
        out[hex_ids] = w_hex * val

    return out
#用
def feet_air_time_type_weighted(
    env,
    command_name: str,
    # biped/quad 各自阈值 + 内部权重
    threshold_biped: float,
    threshold_quad: float,
    threshold_hex: float,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    # biped/quad 各自传感器（以及脚 body_ids）
    biped_sensor_cfg: SceneEntityCfg = SceneEntityCfg("biped_contact_forces"),
    quad_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("quad_contact_forces"),
    hex_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod_contact_forces"),
    # 维持你原逻辑：cmd 很小则 reward=0
    cmd_min_norm: float = 0.1,
) -> torch.Tensor:
    """
    Reward long steps taken by the feet using air-time (type-weighted for biped & quad).

    - For biped envs: uses biped sensor, biped body_ids, threshold_biped, scaled by w_biped
    - For quad envs:  uses quad  sensor, quad  body_ids, threshold_quad,  scaled by w_quad
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # no reward for zero command (same as original)
    cmd = env.command_manager.get_command(command_name)
    move_mask = (torch.norm(cmd[:, :2], dim=1) > cmd_min_norm).float()

    if biped_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        first_contact = cs.compute_first_contact(env.step_dt)[biped_ids][:, biped_sensor_cfg.body_ids]
        last_air_time = cs.data.last_air_time[biped_ids][:, biped_sensor_cfg.body_ids]
        rew = torch.sum((last_air_time - threshold_biped) * first_contact, dim=1)
        out[biped_ids] = w_biped * rew * move_mask[biped_ids]

    if quad_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        first_contact = cs.compute_first_contact(env.step_dt)[quad_ids][:, quad_sensor_cfg.body_ids]
        last_air_time = cs.data.last_air_time[quad_ids][:, quad_sensor_cfg.body_ids]
        rew = torch.sum((last_air_time - threshold_quad) * first_contact, dim=1)
        out[quad_ids] = w_quad * rew * move_mask[quad_ids]

    if hex_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        first_contact = cs.compute_first_contact(env.step_dt)[hex_ids][:, hex_sensor_cfg.body_ids]
        last_air_time = cs.data.last_air_time[hex_ids][:, hex_sensor_cfg.body_ids]
        rew = torch.sum((last_air_time - threshold_hex) * first_contact, dim=1)
        out[hex_ids] = w_hex * rew * move_mask[hex_ids]

    return out

def feet_stumble_type_weighted(
    env,
    # biped/quad 各自 sensor + 阈值尺度 + 内部权重
    biped_sensor_cfg: SceneEntityCfg,
    quad_sensor_cfg:  SceneEntityCfg,
    hex_sensor_cfg:  SceneEntityCfg,
    scale_biped: float = 5.0,
    scale_quad: float = 5.0,
    scale_hex: float = 5.0,
    w_biped: float = -1.0,
    w_quad: float = -1.0,
    w_hex: float = -1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalize feet stumbling (type-weighted for biped & quad).

    原逻辑等价：判断是否存在某个时刻/某只脚满足
        ||F|| > scale * |Fz|
    触发则计数（按脚维度统计），最后对每个 env 求和得到 penalty。
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    def _stumble_penalty(cs: ContactSensor, ids: torch.Tensor, cfg: SceneEntityCfg, scale: float) -> torch.Tensor:
        # 取出该类型 env 的历史接触力: [N,H,F,3]
        f = cs.data.net_forces_w_history[ids][:, :, cfg.body_ids, :]
        f_norm = torch.norm(f, dim=-1)                        # [N,H,F]
        fz_abs = torch.abs(f[..., 2]) + eps                   # [N,H,F]
        stumble = f_norm > (scale * fz_abs)                   # [N,H,F]
        # 对 history 维(any) -> [N,F]，再对脚求和 -> [N]
        return torch.any(stumble, dim=1).float().sum(dim=1)

    if biped_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        out[biped_ids] = w_biped * _stumble_penalty(cs, biped_ids, biped_sensor_cfg, scale_biped)

    if quad_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        out[quad_ids] = w_quad * _stumble_penalty(cs, quad_ids, quad_sensor_cfg, scale_quad)

    if hex_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        out[hex_ids] = w_hex * _stumble_penalty(cs, hex_ids, hex_sensor_cfg, scale_hex)

    return out

def air_time_variance_penalty_type_weighted(
    env,
    w_biped: float = 1.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    biped_sensor_cfg: SceneEntityCfg = SceneEntityCfg("biped_contact_forces"),
    quad_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("quad_contact_forces"),
    hex_sensor_cfg:  SceneEntityCfg = SceneEntityCfg("hexapod_contact_forces"),
    clip_max: float = 0.5,
) -> torch.Tensor:
    """
    Type-weighted penalty on variance of air-time and contact-time across feet.

    Returns >=0; typically use a NEGATIVE RewTerm.weight to penalize variance.
    """
    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    def _pen(cs: ContactSensor, ids: torch.Tensor, cfg: SceneEntityCfg) -> torch.Tensor:
        if cs.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        air = cs.data.last_air_time[ids][:, cfg.body_ids]
        contact = cs.data.last_contact_time[ids][:, cfg.body_ids]
        return torch.var(torch.clip(air, max=clip_max), dim=1) + torch.var(torch.clip(contact, max=clip_max), dim=1)

    if biped_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        out[biped_ids] = w_biped * _pen(cs, biped_ids, biped_sensor_cfg)

    if quad_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        out[quad_ids] = w_quad * _pen(cs, quad_ids, quad_sensor_cfg)

    if hex_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        out[hex_ids] = w_hex * _pen(cs, hex_ids, hex_sensor_cfg)

    return out


"""
Foot gait rewards.
"""

#用
def feet_gait_type_weighted(
    env:MultiLocoEnv,    
    biped_sensor_cfg: SceneEntityCfg,
    quad_sensor_cfg: SceneEntityCfg,
    hex_sensor_cfg: SceneEntityCfg,
    # biped gait params
    period_biped: float,
    offset_biped: list[float],   # len=2
    threshold_biped: float,
    w_biped: float,
    # quad gait params
    period_quad: float,
    offset_quad: list[float],    # len=4
    threshold_quad: float,
    w_quad: float,
    # hex gait params
    period_hex: float,
    offset_hex: list[float],    # len=6
    threshold_hex: float,
    w_hex: float,
    # speed command gating
    command_name: str | None = None,
    cmd_min_norm: float = 0.01,
    # sensors (body_names -> body_ids 由 manager resolve)

) -> torch.Tensor:
    device = env.device
    out = torch.zeros(env.num_envs, dtype=torch.float32, device=device)

    biped_ids, quad_ids, hex_ids = _active_ids(env)

    # --------- common: global phase (per env) ----------
    # shape: (N,1)
    t = (env.episode_length_buf * env.step_dt).unsqueeze(1)

    # --------- biped ----------
    if biped_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        is_contact = cs.data.current_contact_time[biped_ids][:, biped_sensor_cfg.body_ids] > 0  # (Nb,2)

        # global phase in [0,1)
        global_phase = torch.remainder(t[biped_ids], period_biped) / period_biped  # (Nb,1)

        # leg_phase: (Nb,2)
        phases = []
        for off in offset_biped:
            phases.append(torch.remainder(global_phase + off, 1.0))
        leg_phase = torch.cat(phases, dim=-1)

        # reward: count matches between stance/contact
        r = torch.zeros(biped_ids.numel(), device=device, dtype=torch.float32)
        for i in range(len(biped_sensor_cfg.body_ids)):  # 2
            is_stance = leg_phase[:, i] < threshold_biped
            r += (~(is_stance ^ is_contact[:, i])).to(torch.float32)

        out[biped_ids] = w_biped * r

    # --------- quad ----------
    if quad_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        is_contact = cs.data.current_contact_time[quad_ids][:, quad_sensor_cfg.body_ids] > 0  # (Nq,4)

        global_phase = torch.remainder(t[quad_ids], period_quad) / period_quad  # (Nq,1)

        phases = []
        for off in offset_quad:
            phases.append(torch.remainder(global_phase + off, 1.0))
        leg_phase = torch.cat(phases, dim=-1)  # (Nq,4)

        r = torch.zeros(quad_ids.numel(), device=device, dtype=torch.float32)
        for i in range(len(quad_sensor_cfg.body_ids)):  # 4
            is_stance = leg_phase[:, i] < threshold_quad
            r += (~(is_stance ^ is_contact[:, i])).to(torch.float32)

        out[quad_ids] = w_quad * r

    # --------- hex ----------
    if hex_ids.numel() > 0:
        cs: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        is_contact = cs.data.current_contact_time[hex_ids][:, hex_sensor_cfg.body_ids] > 0  # (Nh,6)

        global_phase = torch.remainder(t[hex_ids], period_hex) / period_hex  # (Nh,1)

        phases = []
        for off in offset_hex:
            phases.append(torch.remainder(global_phase + off, 1.0))
        leg_phase = torch.cat(phases, dim=-1)  # (Nh,6)

        r = torch.zeros(hex_ids.numel(), device=device, dtype=torch.float32)
        for i in range(len(hex_sensor_cfg.body_ids)):  # 6
            is_stance = leg_phase[:, i] < threshold_hex
            r += (~(is_stance ^ is_contact[:, i])).to(torch.float32)

        out[hex_ids] = w_hex * r

    # --------- speed command gating ----------
    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)  # (N,3) typically
        cmd_norm = torch.norm(cmd, dim=1)
        out = out * (cmd_norm > cmd_min_norm).to(out.dtype)

    return out

def trot_typed_weight(
    env,
    biped_asset_cfg: SceneEntityCfg,
    quad_asset_cfg: SceneEntityCfg,
    hex_asset_cfg: SceneEntityCfg,
    biped_sensor_cfg: SceneEntityCfg,
    quad_sensor_cfg: SceneEntityCfg,
    hex_sensor_cfg: SceneEntityCfg,
    w_biped: float = 0.0,
    w_quad: float = 1.0,
    w_hex: float = 1.0,
    command_name: str = "base_velocity",    
) -> torch.Tensor:
    # 速度相关权重
    commands = env.command_manager.get_command(command_name)
    scale = 0.2 * torch.clip(torch.abs(commands[:,0]) / (torch.norm(commands[:,1:3],dim=-1) + 0.001), 0.01, 1)

    biped_ids, quad_ids, hex_ids = _active_ids(env)
    out = torch.zeros(env.num_envs, device=env.device)

    if biped_ids.numel() > 0:
        contact_sensor_biped: ContactSensor = env.scene.sensors[biped_sensor_cfg.name]
        contacts_biped = contact_sensor_biped.data.net_forces_w[biped_ids][:, biped_sensor_cfg.body_ids, 2] > 0.1  # (Nb,2)
        contact_results_biped = torch.logical_not(torch.logical_xor(contacts_biped[:,(0,1)],contacts_biped[:,(1,0)]))
        contact_results_biped = torch.sum(contact_results_biped, dim=1)

        asset_biped: Articulation = env.scene[biped_asset_cfg.name]
        dof_pos_biped = asset_biped.data.joint_pos[biped_ids]
        dof_results_biped = torch.sum(torch.square(dof_pos_biped[:,1:3] - dof_pos_biped[:,5:6]),dim=1)

        out[biped_ids] = w_biped * (contact_results_biped + dof_results_biped) * scale[biped_ids]

    if quad_ids.numel() > 0:
        contact_sensor_quad: ContactSensor = env.scene.sensors[quad_sensor_cfg.name]
        contacts_quad = contact_sensor_quad.data.net_forces_w[quad_ids][:, quad_sensor_cfg.body_ids, 2] > 0.1  # (Nq,4)
        contact_results_quad = torch.logical_xor(contacts_quad[:,(0,1,2,3)],contacts_quad[:,(3, 2, 1, 0)]) + \
                               torch.logical_not(torch.logical_xor(contacts_quad[:,(0,1,2,3)],contacts_quad[:,(1, 0, 3, 2)]))
        contact_results_quad = torch.sum(contact_results_quad, dim=1)

        asset_quad: Articulation = env.scene[quad_asset_cfg.name]
        dof_pos_quad = asset_quad.data.joint_pos[quad_ids]      

        dof_results1_quad = torch.sum(torch.square((dof_pos_quad[:,1:3] - dof_pos_quad[:,10:12])),dim=1)
        dof_results2_quad = torch.sum(torch.square((dof_pos_quad[:,4:6] - dof_pos_quad[:,7:9])),dim=1)
        dof_results_quad = dof_results1_quad + dof_results2_quad

        out[quad_ids] = w_quad * (contact_results_quad + dof_results_quad) * scale[quad_ids]

    if hex_ids.numel() > 0:
        contact_sensor_hex: ContactSensor = env.scene.sensors[hex_sensor_cfg.name]
        contacts_hex = contact_sensor_hex.data.net_forces_w[hex_ids][:, hex_sensor_cfg.body_ids, 2] > 0.1
        contact_results_hex = torch.sum(contacts_hex, dim=1)

        asset_hex: Articulation = env.scene[hex_asset_cfg.name]
        dof_pos_hex = asset_hex.data.joint_pos[hex_ids]
        dof_results_hex = torch.sum(torch.square(dof_pos_hex[:, 1:3] - dof_pos_hex[:, 16:18]), dim=1)

        out[hex_ids] = w_hex * (contact_results_hex + dof_results_hex) * scale[hex_ids]

    return out

def trot_phase(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    # raise not NotImplementedError
    commands = env.command_manager.get_command(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1
    contact_results = torch.logical_xor(contacts[:,(0,1,2,3)],contacts[:,(3, 2, 1, 0)]) + \
                               torch.logical_not(torch.logical_xor(contacts[:,(0,1,2,3)],contacts[:,(1, 0, 3, 2)]))
    contact_results = torch.sum(contact_results, dim=1) * (0.2 * torch.clip(torch.abs(commands[:,0]) / (torch.norm(commands[:,1:3],dim=-1) + 0.001), 0.01, 1))

    # 关节角度相同

    asset: Articulation = env.scene[asset_cfg.name]

    dof_pos = asset.data.joint_pos
    dof_results1 = torch.sum(torch.square((dof_pos[:,1:3] - dof_pos[:,10:12])),dim=1)
    dof_results2 = torch.sum(torch.square((dof_pos[:,4:6] - dof_pos[:,7:9])),dim=1)
    dof_results = dof_results1 + dof_results2
    dof_results = dof_results * (0.2 * torch.clip(torch.abs(commands[:,0]) / (torch.norm(commands[:,1:3],dim=-1) + 0.001), 0.01, 1))
    return contact_results + dof_results
