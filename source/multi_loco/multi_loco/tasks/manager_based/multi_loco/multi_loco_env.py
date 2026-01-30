import torch

from isaaclab.envs import ManagerBasedRLEnv
from .multi_loco_env_cfg import MultiLocoEnvCfg
from isaaclab.envs.common import VecEnvStepReturn

class MultiLocoEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: MultiLocoEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # 注意：这里不要再 init env_type 了（会太晚）

    def load_managers(self):
        # 这个函数在 ManagerBasedEnv.__init__ 里被调用
        # 这里 env.num_envs 和 env.device 一般已经可用
        if not hasattr(self, "env_type") or self.env_type is None:
            self.env_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            if self.num_envs >= 3:
                split = self.num_envs // 3
                self.env_type[split : 2 * split] = 1
                self.env_type[2 * split :] = 2
            else:
                self.env_type[self.num_envs // 2 :] = 1

        if not hasattr(self, "act_mask") or self.act_mask is None:
            action_dim = 18
            self.act_mask = torch.zeros((self.num_envs, action_dim), device=self.device, dtype=torch.float32)
            biped_ids = torch.nonzero(self.env_type == 0, as_tuple=False).squeeze(-1)
            quad_ids  = torch.nonzero(self.env_type == 1, as_tuple=False).squeeze(-1)
            hex_ids  = torch.nonzero(self.env_type == 2, as_tuple=False).squeeze(-1)
            self.act_mask[biped_ids, :6] = 1.0
            self.act_mask[quad_ids, :12] = 1.0
            self.act_mask[hex_ids, :18] = 1.0
        # if not hasattr(self, "_dbg_mask_once"):
        #     self._dbg_mask_once = True
        #     print("env_type_dist:", torch.unique(self.env_type, return_counts=True))
        #     print("act_mask[0]:", self.act_mask[0].detach().cpu().tolist())

        # 最后再让父类去创建 ObservationManager / ActionManager / ...
        super().load_managers()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs


        # # -- update command    调到了reward的前面   这是因为 command 可能影响 reward 计算  GPT改的
        # self.command_manager.compute(dt=self.step_dt)
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # asset = self.scene["biped"]  # 或你实际 active 的那只
        # if self.common_step_count < 20:
        #     print("q_target std:", asset.data.joint_pos_target[0].std().item())

        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
