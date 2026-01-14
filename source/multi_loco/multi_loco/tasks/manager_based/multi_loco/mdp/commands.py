from __future__ import annotations

import torch
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from ..multi_loco_env import MultiLocoEnv

@configclass
class MultiRobotUniformVelocityCommandCfg(CommandTermCfg):
    """像 UniformVelocityCommandCfg 一样，但支持 biped/quad 两个 asset，通过 env.env_type 选择激活者。"""

    class_type: type = MISSING  # 下面会填成 MultiRobotUniformVelocityCommand

    # 两个机器人在 scene 里的名字
    biped_asset_name: str = "biped"
    quad_asset_name: str = "quad"

    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING

    # debug 可视化配置（跟原版一致）
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


class MultiRobotUniformVelocityCommand(CommandTerm):
    cfg: "MultiRobotUniformVelocityCommandCfg"

    def __init__(self, cfg: "MultiRobotUniformVelocityCommandCfg", env: "MultiLocoEnv"):
        super().__init__(cfg, env)

        # check configuration（照抄原版）
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError("heading_command=True but ranges.heading is None.")
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn("ranges.heading is set but heading_command is False.")

        # two robot assets
        self.biped: Articulation = env.scene[cfg.biped_asset_name]
        self.quad: Articulation = env.scene[cfg.quad_asset_name]

        # command buffers（照抄原版）
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        # metrics（照抄原版）
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    # --------- 核心：根据 env.env_type 取 “激活机器人” 数据 ---------
    def _active_data(self):
        """返回 active robot 的数据视图（按 env.env_type 拼起来）"""
        env = self._env
        device = self.device
        env_ids = torch.arange(self.num_envs, device=device)
        biped_ids = env_ids[env.env_type == 0]
        quad_ids  = env_ids[env.env_type == 1]

        root_lin_vel_b = torch.zeros((self.num_envs, 3), device=device)
        root_ang_vel_b = torch.zeros((self.num_envs, 3), device=device)
        heading_w      = torch.zeros((self.num_envs,), device=device)
        root_pos_w     = torch.zeros((self.num_envs, 3), device=device)
        root_quat_w    = torch.zeros((self.num_envs, 4), device=device)

        if biped_ids.numel() > 0:
            root_lin_vel_b[biped_ids] = self.biped.data.root_lin_vel_b[biped_ids]
            root_ang_vel_b[biped_ids] = self.biped.data.root_ang_vel_b[biped_ids]
            heading_w[biped_ids]      = self.biped.data.heading_w[biped_ids]
            root_pos_w[biped_ids]     = self.biped.data.root_pos_w[biped_ids]
            root_quat_w[biped_ids]    = self.biped.data.root_quat_w[biped_ids]

        if quad_ids.numel() > 0:
            root_lin_vel_b[quad_ids] = self.quad.data.root_lin_vel_b[quad_ids]
            root_ang_vel_b[quad_ids] = self.quad.data.root_ang_vel_b[quad_ids]
            heading_w[quad_ids]      = self.quad.data.heading_w[quad_ids]
            root_pos_w[quad_ids]     = self.quad.data.root_pos_w[quad_ids]
            root_quat_w[quad_ids]    = self.quad.data.root_quat_w[quad_ids]

        return root_lin_vel_b, root_ang_vel_b, heading_w, root_pos_w, root_quat_w

    # --------- 跟原版一样：metrics、resample、update、debug_vis ---------
    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        root_lin_vel_b, root_ang_vel_b, _, _, _ = self._active_data()

        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            _, _, heading_w, _, _ = self._active_data()

            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # 两个机器人都可能存在 init 问题，保守一点：只要 active 的那台能读数据就行
        # 这里简单判断：如果两台都没初始化就返回
        if (not self.biped.is_initialized) and (not self.quad.is_initialized):
            return

        root_lin_vel_b, _, _, root_pos_w, root_quat_w = self._active_data()

        base_pos_w = root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2], root_quat_w)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(root_lin_vel_b[:, :2], root_quat_w)

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, base_quat_w: torch.Tensor):
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = MISSING  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""


class UniformGaitCommand(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration."""

    cfg: UniformGaitCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformGaitCommandCfg, env: MultiLocoEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        # command format: [frequency, phase offset, contact duration]
        self.gait_command = torch.zeros(self.num_envs, 4, device=self.device)
        # self.gait_command = torch.zeros(self.num_envs, 3, device=self.device)
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 3)."""
        return self.gait_command

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments."""
        # sample gait parameters
        r = torch.empty(len(env_ids), device=self.device)
        # -- frequency
        self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
        # -- phase offset
        self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.offsets)
        # -- contact duration
        self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.durations)
        # -- swing height
        self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.swing_height)

    def _update_command(self):
        """Update the command. No additional processing needed in this implementation."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass
