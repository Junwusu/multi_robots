from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.log
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from ..multi_loco_env import MultiLocoEnv

class MultiRobotJointAction(ActionTerm):
    """Base action term: policy action_dim is fixed to 12, then we scale/offset/clip it."""

    cfg: "MultiRobotJointActionCfg"

    def __init__(self, cfg: "MultiRobotJointActionCfg", env: "MultiLocoEnv") -> None:
        super().__init__(cfg, env)

        # resolve assets
        self._biped: Articulation = env.scene[cfg.biped_cfg.name]
        self._quad: Articulation = env.scene[cfg.quad_cfg.name]

        # resolve joints for each asset
        self._biped_joint_ids, self._biped_joint_names = self._biped.find_joints(
            cfg.biped_cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._quad_joint_ids, self._quad_joint_names = self._quad.find_joints(
            cfg.quad_cfg.joint_names, preserve_order=cfg.preserve_order
        )

        if len(self._biped_joint_ids) != 6:
            omni.log.warn(f"[MultiRobotJointAction] biped joints resolved = {len(self._biped_joint_ids)} (expected 6).")
        if len(self._quad_joint_ids) != 12:
            omni.log.warn(f"[MultiRobotJointAction] quad joints resolved = {len(self._quad_joint_ids)} (expected 12).")

        omni.log.info(
            f"Resolved biped joints: {self._biped_joint_names} [{self._biped_joint_ids}] | "
            f"quad joints: {self._quad_joint_names} [{self._quad_joint_ids}]"
        )

        # raw/processed action buffers: always (num_envs, 12)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # parse scale/offset/clip for 12 dims (simple + robust)
        # We keep these as either float or (num_envs, 12) tensors.
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # dict keys are joint names patterns; we support patterns over a "virtual" 12-dim name list
            # Provide names in order: biped6 + quad12? No: action is universal 12, so define indices explicitly.
            # Easiest: allow dict keys like "a0".."a11". (Or just use float scale.)
            idx, _, val = string_utils.resolve_matching_names_values(cfg.scale, [f"a{i}" for i in range(12)])
            self._scale[:, idx] = torch.tensor(val, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            idx, _, val = string_utils.resolve_matching_names_values(cfg.offset, [f"a{i}" for i in range(12)])
            self._offset[:, idx] = torch.tensor(val, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}")

        if cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                idx, _, val = string_utils.resolve_matching_names_values(cfg.clip, [f"a{i}" for i in range(12)])
                self._clip[:, idx] = torch.tensor(val, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")
        else:
            self._clip = None
        
        # print("asset_name:", cfg.asset_name, "biped:", cfg.biped_cfg.name, "quad:", cfg.quad_cfg.name,"scene keys:", list(env.scene.keys()))


    @property
    def action_dim(self) -> int:
        # IMPORTANT: policy always outputs 12
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids=None) -> None:
        self._raw_actions[env_ids] = 0.0


class MultiRobotJointPositionAction(MultiRobotJointAction):
    """Apply masked 12-dim actions to biped(6) + quad(12) as position targets."""

    cfg: "MultiRobotJointPositionActionCfg"

    def __init__(self, cfg: "MultiRobotJointPositionActionCfg", env: "MultiLocoEnv"):
        super().__init__(cfg, env)

        # optional: use default joint positions as offset (like JointPositionAction)
        # Here we interpret "default offset" as per-asset default joint positions,
        # then we add processed actions as delta.
        self._use_default_offset = cfg.use_default_offset

        if self._use_default_offset:
            # Store per-asset default joint pos (num_envs, dof)
            # NOTE: these tensors exist after assets are initialized.
            self._biped_default = self._biped.data.default_joint_pos[:, self._biped_joint_ids].clone()
            self._quad_default = self._quad.data.default_joint_pos[:, self._quad_joint_ids].clone()

    def apply_actions(self):
        env = self._env
        device = self.device

        # 1) mask (num_envs, 12). For biped: last 6 -> 0
        # Expect env.act_mask exists and is float (num_envs, 12)
        masked = self.processed_actions * env.act_mask

        # 2) split env ids
        env_ids = torch.arange(env.num_envs, device=device)
        biped_ids = env_ids[env.env_type == 0]
        quad_ids  = env_ids[env.env_type == 1]

        # 3) build targets
        if biped_ids.numel() > 0:
            biped_delta = masked[biped_ids, :6]  # (Nb, 6)
            if self._use_default_offset:
                biped_target = self._biped_default[biped_ids] + biped_delta
            else:
                biped_target = biped_delta
            self._biped.set_joint_position_target(biped_target, joint_ids=self._biped_joint_ids, env_ids=biped_ids)

            # if not hasattr(env, "_dbg_cnt"):
            #     env._dbg_cnt = 0
            # if env._dbg_cnt < 10 and biped_ids.numel() > 0:
            #     env._dbg_cnt += 1
            #     jp = self._biped.data.joint_pos[biped_ids[0], self._biped_joint_ids]
            #     print("[DBG] delta_std:", biped_delta[0].std().item(),
            #         "tgt_std:", biped_target[0].std().item(),
            #         "pos_std:", jp.std().item())
            #     print("[DBG] delta:", biped_delta[0].detach().cpu().numpy())
            #     print("[DBG] tgt  :", biped_target[0].detach().cpu().numpy())
            #     print("[DBG] pos  :", jp.detach().cpu().numpy())



        if quad_ids.numel() > 0:
            quad_delta = masked[quad_ids, :12]   # (Nq, 12)
            if self._use_default_offset:
                quad_target = self._quad_default[quad_ids] + quad_delta
            else:
                quad_target = quad_delta
            self._quad.set_joint_position_target(quad_target, joint_ids=self._quad_joint_ids, env_ids=quad_ids)


@configclass
class MultiRobotJointActionCfg(ActionTermCfg):
    """Cfg: universal 12-dim joint action over two assets."""

    biped_cfg: SceneEntityCfg = MISSING
    quad_cfg: SceneEntityCfg = MISSING

    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    clip: dict[str, tuple[float, float]] | None = None
    preserve_order: bool = False


@configclass
class MultiRobotJointPositionActionCfg(MultiRobotJointActionCfg):
    """Cfg for position action."""

    class_type: type[ActionTerm] = MultiRobotJointPositionAction
    use_default_offset: bool = True
