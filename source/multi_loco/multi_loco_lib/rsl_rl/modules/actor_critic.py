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

class MorphGNNEncoder(nn.Module):
    """
    Super-graph GNN for N=12 joints (nodes).
    Node features: [mask, q, qd]  ->  z_morph [B, D]
    """
    def __init__(
        self,
        num_nodes: int = 12,
        node_feat_dim: int = 3,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        act_fn: nn.Module = nn.ELU(),
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.node_feat_dim = int(node_feat_dim)
        self.embed_dim = int(embed_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.node_feat_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, self.embed_dim),
            act_fn,
        )

        # Default adjacency: chain + self-loops (N fixed)
        A = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        for i in range(self.num_nodes):
            A[i, i] = 1.0
            if i - 1 >= 0:
                A[i, i - 1] = 1.0
            if i + 1 < self.num_nodes:
                A[i, i + 1] = 1.0
        deg = A.sum(dim=1, keepdim=True).clamp_min(1.0)
        A_norm = A / deg
        self.register_buffer("A_norm", A_norm, persistent=False)  # [N, N]

    def forward(self, node_feat: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        node_feat: [B, N, F]
        node_mask: [B, N]  (0/1)
        returns z: [B, D]
        """
        # mask invalid nodes (both features and pooling)
        m = node_mask.unsqueeze(-1)  # [B,N,1]
        x = node_feat * m

        h = self.mlp1(x)  # [B,N,H]
        # neighbor mean: A_norm @ h
        h_nb = torch.einsum("ij,bjh->bih", self.A_norm, h)
        h = h + h_nb
        h = self.mlp2(h)  # [B,N,D]

        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        z = (h * m).sum(dim=1) / denom  # [B,D]
        return z

class ActorCriticMultiCritic_GNN(nn.Module):
    """
    - One actor
    - Two critics (biped/quad), BUT value is soft-mixture via learned gate(z_morph)
    - No env_type/onehot is used. Morphology is inferred from (act_mask + q + qd).
    """
    is_recurrent = False
    is_sequence = False
    is_vae = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        orthogonal_init: bool = False,
        init_noise_std: float = 1.0,
        # log_std clamp (paper settings)
        log_std_min: float = -20.0,
        log_std_max: float = 4.0,
        # --- GNN configs ---
        use_morph_gnn: bool = True,
        morph_embed_dim: int = 32,
        num_joints_super: int = 12,
        gnn_hidden_dim: int = 128,
        # --- obs layout: based on your current cfg order ---
        # Policy obs order:
        # base_ang_vel(3), proj_gravity(3),
        # joint_pos(12), joint_vel(12), last_action(12),
        # gait_phase(2), gait_command(4),
        # act_mask(12)  <-- you added and should keep as LAST
        actor_joint_pos_start: int = 6,
        actor_joint_vel_start: int = 18,
        actor_act_mask_is_last_12: bool = True,
        # Critic obs order:
        # base_lin_vel(3), base_ang_vel(3), proj_gravity(3),
        # joint_pos(12), joint_vel(12), last_action(12),
        # gait_phase(2), gait_command(4),
        # act_mask(12)  <-- keep as LAST
        critic_joint_pos_start: int = 9,
        critic_joint_vel_start: int = 21,
        critic_act_mask_is_last_12: bool = True,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMultiCritic_GNN.__init__ got unexpected arguments, ignored:",
                list(kwargs.keys()),
            )
        super().__init__()

        act_fn = get_activation(activation)

        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.orthogonal_init = bool(orthogonal_init)

        self.use_morph_gnn = bool(use_morph_gnn)
        self.morph_embed_dim = int(morph_embed_dim)
        self.num_joints_super = int(num_joints_super)

        # ---- slices (actor) ----
        self.actor_joint_pos_slice = slice(actor_joint_pos_start, actor_joint_pos_start + self.num_joints_super)
        self.actor_joint_vel_slice = slice(actor_joint_vel_start, actor_joint_vel_start + self.num_joints_super)
        if actor_act_mask_is_last_12:
            self.actor_mask_slice = slice(self.num_actor_obs - self.num_joints_super, self.num_actor_obs)
        else:
            raise ValueError("actor_act_mask_is_last_12=False is not supported in this drop-in version.")

        # ---- slices (critic) ----
        self.critic_joint_pos_slice = slice(critic_joint_pos_start, critic_joint_pos_start + self.num_joints_super)
        self.critic_joint_vel_slice = slice(critic_joint_vel_start, critic_joint_vel_start + self.num_joints_super)
        if critic_act_mask_is_last_12:
            self.critic_mask_slice = slice(self.num_critic_obs - self.num_joints_super, self.num_critic_obs)
        else:
            raise ValueError("critic_act_mask_is_last_12=False is not supported in this drop-in version.")

        # ---- build networks with augmented dims ----
        actor_in_dim = self.num_actor_obs + (self.morph_embed_dim if self.use_morph_gnn else 0)
        critic_in_dim = self.num_critic_obs + (self.morph_embed_dim if self.use_morph_gnn else 0)

        self.actor = self._build_mlp(
            actor_in_dim, self.num_actions, actor_hidden_dims, act_fn, self.orthogonal_init, out_gain=0.01
        )
        self.critic_biped = self._build_mlp(
            critic_in_dim, 1, critic_hidden_dims, act_fn, self.orthogonal_init, out_gain=0.01
        )
        self.critic_quad = self._build_mlp(
            critic_in_dim, 1, critic_hidden_dims, act_fn, self.orthogonal_init, out_gain=0.01
        )

        # ---- morph gnn + gate ----
        if self.use_morph_gnn:
            # node_feat = [mask, q, qd] => dim=3
            self.morph_gnn = MorphGNNEncoder(
                num_nodes=self.num_joints_super,
                node_feat_dim=3,
                embed_dim=self.morph_embed_dim,
                hidden_dim=int(gnn_hidden_dim),
                act_fn=act_fn,
            )
            self.gate = nn.Sequential(
                nn.Linear(self.morph_embed_dim, 64),
                act_fn,
                nn.Linear(64, 1),  # will apply sigmoid in evaluate()
            )
        else:
            self.morph_gnn = None
            self.gate = None

        # ---- action noise ----
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        init_logstd = float(np.log(init_noise_std))
        self.logstd = nn.Parameter(torch.full((self.num_actions,), init_logstd, dtype=torch.float32))

        self.distribution = None
        Normal.set_default_validate_args = False

        print(
            "[ActorCriticMultiCritic_GNN] init:",
            f"num_actor_obs={self.num_actor_obs}, num_critic_obs={self.num_critic_obs}, num_actions={self.num_actions}",
            f"use_morph_gnn={self.use_morph_gnn}, morph_embed_dim={self.morph_embed_dim}",
        )
        print(
            "[ActorCriticMultiCritic_GNN] slices:",
            f"actor_joint_pos={self.actor_joint_pos_slice}, actor_joint_vel={self.actor_joint_vel_slice}, actor_mask={self.actor_mask_slice}",
            f"critic_joint_pos={self.critic_joint_pos_slice}, critic_joint_vel={self.critic_joint_vel_slice}, critic_mask={self.critic_mask_slice}",
        )
        print(f"[ActorCriticMultiCritic_GNN] logstd_init={init_logstd:.3f}, clamp=[{self.log_std_min},{self.log_std_max}]")
        print("[DEBUG] init logstd =", self.logstd[:4].detach().cpu().numpy())

        # one-time debug flag
        self._dbg_once = False

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

    # ---------------- Morphology embedding ----------------
    def _compute_z_from_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        mask = obs[:, self.actor_mask_slice]  # [B,12]
        q = obs[:, self.actor_joint_pos_slice]  # [B,12]
        qd = obs[:, self.actor_joint_vel_slice]  # [B,12]
        node_feat = torch.stack([mask, q, qd], dim=-1)  # [B,12,3]
        node_mask = (mask > 0.5).float()  # [B,12]
        z = self.morph_gnn(node_feat, node_mask=node_mask)  # [B,D]
        return z

    def _compute_z_from_critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        mask = obs[:, self.critic_mask_slice]
        q = obs[:, self.critic_joint_pos_slice]
        qd = obs[:, self.critic_joint_vel_slice]
        node_feat = torch.stack([mask, q, qd], dim=-1)
        node_mask = (mask > 0.5).float()
        z = self.morph_gnn(node_feat, node_mask=node_mask)
        return z

    def _augment_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.use_morph_gnn:
            return obs
        z = self._compute_z_from_actor_obs(obs)
        return torch.cat([obs, z], dim=-1)

    def _augment_critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.use_morph_gnn:
            return obs
        z = self._compute_z_from_critic_obs(obs)
        return torch.cat([obs, z], dim=-1)

    # ---------------- Actor ----------------
    def update_distribution(self, observations: torch.Tensor):
        if (not self._dbg_once) and self.use_morph_gnn:
            self._dbg_once = True
            with torch.no_grad():
                m0 = observations[0, self.actor_mask_slice].detach().cpu()
                print("[DEBUG] actor mask[0] =", m0.tolist())
                # quick sanity check: first env should be biped -> last 6 ~ 0
                print("[DEBUG] actor q[0][:6] =", observations[0, self.actor_joint_pos_slice][:6].detach().cpu().tolist())

        obs_aug = self._augment_actor_obs(observations)
        mean = self.actor(obs_aug)

        logstd = torch.clamp(self.logstd, self.log_std_min, self.log_std_max).to(mean.device)
        std = torch.exp(logstd).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        obs_aug = self._augment_actor_obs(observations)
        return self.actor(obs_aug)

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ---------------- Critic (two critics + learned soft gate) ----------------
    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        critic_aug = self._augment_critic_obs(critic_observations)

        if self.use_morph_gnn:
            z = critic_aug[:, -self.morph_embed_dim:]  # [B,D]
            g = torch.sigmoid(self.gate(z))  # [B,1]
        else:
            # fallback: constant mix
            g = 0.5

        v_b = self.critic_biped(critic_aug)
        v_q = self.critic_quad(critic_aug)
        v = (1.0 - g) * v_b + g * v_q
        return v