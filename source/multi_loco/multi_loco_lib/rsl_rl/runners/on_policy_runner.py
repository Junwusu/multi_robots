# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import Deque, Tuple

import torch

import rsl_rl
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner as OnPolicyRunnerBase
from rsl_rl.utils import store_code_state
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)

from ..algorithms import PPO
from ..modules import ActorCriticMultiCritic


class OnPolicyRunner(OnPolicyRunnerBase):
    """OnPolicyRunner with mixed-morphology (biped+quad) friendly logging.

    Key features:
    - Does NOT assume env.cfg.actions.joint_pos.joint_names exists.
    - Adds per-env_type episode stats (reward/length).
    - FIX: caches env_type BEFORE env.step() so resets inside step() don't corrupt type attribution.
    - Always prints [TypeStats] every iteration (even if log_dir is None).
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"
            else:
                self.privileged_obs_type = None
        else:  # distillation
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCriticMultiCritic = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            num_rnd_state = rnd_state.shape[1]
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # Per-type rolling buffers
        self._rewbuffer_biped: Deque[float] = deque(maxlen=100)
        self._rewbuffer_quad: Deque[float] = deque(maxlen=100)
        self._lenbuffer_biped: Deque[float] = deque(maxlen=100)
        self._lenbuffer_quad: Deque[float] = deque(maxlen=100)

    # ---------------- helpers ----------------
    def _resolve_base_env(self):
        """Try to resolve the underlying (unwrapped) env where events wrote env_type."""
        e = self.env
        # common wrapper patterns
        if hasattr(e, "unwrapped"):
            try:
                e2 = e.unwrapped
                if e2 is not None:
                    return e2
            except Exception:
                pass
        for name in ["_env", "env", "venv"]:
            if hasattr(e, name):
                e2 = getattr(e, name)
                if e2 is not None:
                    # sometimes nested
                    if hasattr(e2, "unwrapped"):
                        try:
                            return e2.unwrapped
                        except Exception:
                            return e2
                    return e2
        return e

    def _get_env_type(self) -> torch.Tensor:
        """Return env_type [num_envs] on self.device, reading it from the underlying env.
        0=biped, 1=quad. If missing, warn and return a temporary default (DO NOT write back).
        """
        base_env = self._resolve_base_env()

        if hasattr(base_env, "env_type") and getattr(base_env, "env_type") is not None:
            et = getattr(base_env, "env_type")
            if et.device != self.device:
                et = et.to(self.device)
            return et

        print("[WARN] underlying env.env_type not found (events may not have run yet).")
        # temporary default only, do NOT set attributes on wrapper/base
        return torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)



    def _split_done_ids_by_type(self, done_env_ids_1d: torch.Tensor, env_type_snapshot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split done env ids into (biped_done, quad_done) using a snapshot taken BEFORE env.step()."""
        et = env_type_snapshot
        biped_done = done_env_ids_1d[et[done_env_ids_1d] == 0]
        quad_done = done_env_ids_1d[et[done_env_ids_1d] == 1]
        return biped_done, quad_done

    def _log_type_stats(self, it: int):
        """Always print per-type episode stats. Also writes to TB if available."""
        if len(self._rewbuffer_biped) > 0:
            b_rew = statistics.mean(self._rewbuffer_biped)
            b_len = statistics.mean(self._lenbuffer_biped) if len(self._lenbuffer_biped) > 0 else 0.0
        else:
            b_rew, b_len = float("nan"), float("nan")

        if len(self._rewbuffer_quad) > 0:
            q_rew = statistics.mean(self._rewbuffer_quad)
            q_len = statistics.mean(self._lenbuffer_quad) if len(self._lenbuffer_quad) > 0 else 0.0
        else:
            q_rew, q_len = float("nan"), float("nan")

        # also show current env_type distribution (current, not snapshot)
        et_now = self._get_env_type()
        u, c = torch.unique(et_now, return_counts=True)
        dist = dict(zip(u.tolist(), c.tolist()))  # {0: n_biped, 1: n_quad}

        print(
            f"[TypeStats] it={it}  "
            f"biped: rew={b_rew:.3f}, len={b_len:.2f} | "
            f"quad: rew={q_rew:.3f}, len={q_len:.2f} | "
            f"env_type_dist={dist}"
        )

        if self.writer is not None and hasattr(self.writer, "add_scalar"):
            self.writer.add_scalar("type/biped_ep_reward_mean", b_rew, it)
            self.writer.add_scalar("type/biped_ep_len_mean", b_len, it)
            self.writer.add_scalar("type/quad_ep_reward_mean", q_rew, it)
            self.writer.add_scalar("type/quad_ep_len_mean", q_len, it)

    # ---------------- main loop ----------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # FIX: snapshot env_type BEFORE env.step (some envs reset inside step)
                    env_type_prev = self._get_env_type().clone()

                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs)

                    # if it < 2 and _ < 5:  # 前两次迭代的前几步
                    #     print("actions mean/std:", actions.mean().item(), actions.std().item(),
                    #         "min/max:", actions.min().item(), actions.max().item())

                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                    # Move to device
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # normalize
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards

                        # Update episode length
                        cur_episode_length += 1

                        # Clear data for completed episodes
                        new_ids_2d = (dones > 0).nonzero(as_tuple=False)  # shape [k, 1]
                        if new_ids_2d.numel() > 0:
                            done_env_ids = new_ids_2d.squeeze(-1)  # shape [k]

                            # global buffers (original)
                            rewbuffer.extend(cur_reward_sum[done_env_ids].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[done_env_ids].cpu().numpy().tolist())

                            # type buffers (use snapshot BEFORE step)
                            biped_done, quad_done = self._split_done_ids_by_type(done_env_ids, env_type_prev)
                            if biped_done.numel() > 0:
                                self._rewbuffer_biped.extend(cur_reward_sum[biped_done].cpu().numpy().tolist())
                                self._lenbuffer_biped.extend(cur_episode_length[biped_done].cpu().numpy().tolist())
                            if quad_done.numel() > 0:
                                self._rewbuffer_quad.extend(cur_reward_sum[quad_done].cpu().numpy().tolist())
                                self._lenbuffer_quad.extend(cur_episode_length[quad_done].cpu().numpy().tolist())

                            # reset per-episode accumulators
                            cur_reward_sum[done_env_ids] = 0
                            cur_episode_length[done_env_ids] = 0

                            if self.alg.rnd:
                                erewbuffer.extend(cur_ereward_sum[done_env_ids].cpu().numpy().tolist())
                                irewbuffer.extend(cur_ireward_sum[done_env_ids].cpu().numpy().tolist())
                                cur_ereward_sum[done_env_ids] = 0
                                cur_ireward_sum[done_env_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # original logger (kept)
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # ALWAYS print per-type stats
            self._log_type_stats(it)

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter and not self.disable_logs and self.log_dir is not None:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.writer is not None and self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def infer(self, obs, dones=None, extras=None):
        with torch.no_grad():
            if self.empirical_normalization:
                obs = self.obs_normalizer(obs)
            actions = self.alg.policy.act_inference(obs)
        return actions

    def export(self, path, filename="policy.onnx"):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        policy_exporter = _OnnxPolicyExporter(self.alg.policy, self.obs_normalizer)
        policy_exporter.export(path, filename)


import copy


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy: ActorCriticMultiCritic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            # NOTE: if actor is not Sequential, you may need to adapt this line
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
