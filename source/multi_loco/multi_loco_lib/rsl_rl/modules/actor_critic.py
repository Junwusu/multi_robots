# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCritic(nn.Module):
    is_recurrent = False
    is_sequence = False
    is_vae = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        orthogonal_init=False,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCritic, self).__init__()

        self.orthogonal_init = orthogonal_init
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
                actor_layers.append(activation)
                # actor_layers.append(torch.nn.LayerNorm(actor_hidden_dims[l + 1]))
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
                critic_layers.append(activation)
                # critic_layers.append(torch.nn.LayerNorm(critic_hidden_dims[l + 1]))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.logstd = nn.Parameter(torch.zeros(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + torch.exp(self.logstd))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None



# 复用你原来的 get_activation
# from ... import get_activation

class ActorCriticTwoCriticHeads(nn.Module):
    is_recurrent = False
    is_sequence = False
    is_vae = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        orthogonal_init=False,
        init_noise_std=1.0,
        # 关键：告诉 critic robot_type 在 critic_obs 的哪两维
        robot_type_start: int | None = None,   # default: last 2 dims
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTwoCriticHeads.__init__ got unexpected arguments, ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.orthogonal_init = orthogonal_init
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        act_fn = get_activation(activation)

        # ---------------- Policy (shared actor) ----------------
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
        actor_layers.append(act_fn)

        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
                actor_layers.append(act_fn)

        self.actor = nn.Sequential(*actor_layers)

        # ---------------- Critic backbone + two heads ----------------
        # backbone 输出维度 = critic_hidden_dims[-1]
        critic_backbone = []
        critic_backbone.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_backbone.append(act_fn)
        for l in range(len(critic_hidden_dims) - 1):
            critic_backbone.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
            if self.orthogonal_init:
                torch.nn.init.orthogonal_(critic_backbone[-1].weight, np.sqrt(2))
                torch.nn.init.constant_(critic_backbone[-1].bias, 0.0)
            critic_backbone.append(act_fn)

        self.critic_backbone = nn.Sequential(*critic_backbone)

        hid = critic_hidden_dims[-1]
        self.value_head_biped = nn.Linear(hid, 1)
        self.value_head_quad = nn.Linear(hid, 1)

        if self.orthogonal_init:
            torch.nn.init.orthogonal_(self.value_head_biped.weight, 0.01)
            torch.nn.init.constant_(self.value_head_biped.bias, 0.0)
            torch.nn.init.orthogonal_(self.value_head_quad.weight, 0.01)
            torch.nn.init.constant_(self.value_head_quad.bias, 0.0)

        # robot_type onehot slice
        if robot_type_start is None:
            robot_type_start = num_critic_obs - 2
        self.robot_type_slice = slice(robot_type_start, robot_type_start + 2)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic backbone: {self.critic_backbone}")
        print(f"Value heads: biped/quad, robot_type_slice={self.robot_type_slice}")

        # Action noise
        self.logstd = nn.Parameter(torch.zeros(num_actions) * math.log(init_noise_std)  )
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + torch.exp(self.logstd))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        # critic_observations 必须包含 robot_type onehot（最后2维最方便）
        h = self.critic_backbone(critic_observations)
        v_b = self.value_head_biped(h)
        v_q = self.value_head_quad(h)

        rt = critic_observations[:, self.robot_type_slice]  # (N,2) onehot
        is_quad = rt[:, 1] > 0.5
        v = torch.where(is_quad.unsqueeze(-1), v_q, v_b)
        return v



# 复用你已有的 get_activation(act_name)

class ActorCriticMultiCritic(nn.Module):
    is_recurrent = False
    is_sequence = False
    is_vae = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        orthogonal_init=False,
        init_noise_std=1.0,
        # robot_type onehot 在 critic_obs 里的位置（你现在是最后2维）
        robot_type_start: int | None = None,
        # 论文表 7
        log_std_min: float = -20.0,
        log_std_max: float = 4.0,
        **kwargs,
    ):
        if kwargs:
            print("ActorCriticMultiCritic.__init__ got unexpected arguments, ignored:",
                  list(kwargs.keys()))
        super().__init__()

        print("[ActorCriticMultiCritic] init with params:",
              f"num_actor_obs={num_actor_obs}, num_critic_obs={num_critic_obs}, num_actions={num_actions},",
              f"actor_hidden_dims={actor_hidden_dims}, critic_hidden_dims={critic_hidden_dims},",
              f"activation={activation}, orthogonal_init={orthogonal_init}, init_noise_std={init_noise_std},",
              f"robot_type_start={robot_type_start}, log_std_min={log_std_min}, log_std_max={log_std_max}") 

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.orthogonal_init = orthogonal_init
        act_fn = get_activation(activation)

        # ---------------- Actor (shared) ----------------
        self.actor = self._build_mlp(num_actor_obs, num_actions, actor_hidden_dims, act_fn, orthogonal_init, out_gain=0.01)

        # ---------------- Critics (separate nets) ----------------
        self.critic_biped = self._build_mlp(num_critic_obs, 1, critic_hidden_dims, act_fn, orthogonal_init, out_gain=0.01)
        self.critic_quad  = self._build_mlp(num_critic_obs, 1, critic_hidden_dims, act_fn, orthogonal_init, out_gain=0.01)

        # robot_type slice
        if robot_type_start is None:
            robot_type_start = num_critic_obs - 2
        self.robot_type_slice = slice(robot_type_start, robot_type_start + 2)

        # ---------------- Action noise (论文：Std Init=1.0 + clamp logstd) ----------------
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        init_logstd = float(np.log(init_noise_std))
        self.logstd = nn.Parameter(torch.full((num_actions,), init_logstd, dtype=torch.float32))

        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"[ActorCriticMultiCritic] robot_type_slice={self.robot_type_slice}, "
              f"logstd_init={init_logstd:.3f}, clamp=[{self.log_std_min},{self.log_std_max}]")
        
        print("[DEBUG] init logstd =", self.logstd[:4].detach().cpu().numpy())


    def _build_mlp(self, in_dim, out_dim, hidden_dims, act_fn, orthogonal_init, out_gain=0.01):
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        if orthogonal_init:
            nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))
            nn.init.constant_(layers[-1].bias, 0.0)
        layers.append(act_fn)

        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
                if orthogonal_init:
                    nn.init.orthogonal_(layers[-1].weight, out_gain)
                    nn.init.constant_(layers[-1].bias, 0.0)
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                if orthogonal_init:
                    nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))
                    nn.init.constant_(layers[-1].bias, 0.0)
                layers.append(act_fn)
        return nn.Sequential(*layers)

    def reset(self, dones=None):
        # 非 recurrent policy 不需要做任何事，但 rsl_rl 会调用
        return

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)

        # ✅ device 一致 + clamp（修你之前 cuda/cpu 混用风险 + 论文设置）
        logstd = torch.clamp(self.logstd, self.log_std_min, self.log_std_max).to(mean.device)
        std = torch.exp(logstd).expand_as(mean)

        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        # rt onehot: [biped, quad]
        rt = critic_observations[:, self.robot_type_slice]
        is_quad = (rt[:, 1] > 0.5)

        v_b = self.critic_biped(critic_observations)
        v_q = self.critic_quad(critic_observations)

        v = torch.where(is_quad.unsqueeze(-1), v_q, v_b)
        return v

