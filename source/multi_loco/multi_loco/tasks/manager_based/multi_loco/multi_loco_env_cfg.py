# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##
from multi_loco.assets.robots import *  # isort:skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
##
# Scene definition
##


@configclass
class MultiLocoSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robot
    # 双足：名字随便，但后面会用到 asset_name，所以建议清晰
    biped: ArticulationCfg = BRAVER_biped_CFG.replace(
        prim_path="{ENV_REGEX_NS}/BRAVER_BIPED",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38),  # 双足站高一点
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=BRAVER_biped_default_joint_pos,
        ),
    )
    # # 四足
    # quad: ArticulationCfg = BRAVER_quad_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/BRAVER_QUAD",
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.50),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #         joint_pos=BRAVER_QUAD_default_joint_pos,
    #     ),
    # )
    # go1
    quad: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/BRAVER_QUAD")
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # contact sensors
    biped_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/BRAVER_BIPED/.*", history_length=10, track_air_time=True, update_period=0.0,
    )
    quad_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/BRAVER_QUAD/.*", history_length=10, track_air_time=True, update_period=0.0,
    )

##
# MDP settings
##
@configclass
class CommandsCfg:
    base_velocity = mdp.MultiRobotUniformVelocityCommandCfg(
        class_type=mdp.MultiRobotUniformVelocityCommand,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        biped_asset_name="biped",
        quad_asset_name="quad",
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        ranges=mdp.MultiRobotUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi),
        ),
    )
    gait_command = mdp.UniformGaitCommandCfg(
        class_type=mdp.UniformGaitCommand,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(2.5, 2.5),
            offsets=(0.5, 0.5),
            durations=(0.5, 0.5),
            swing_height=(0.10, 0.10),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.MultiRobotJointPositionActionCfg(
        asset_name="biped",
        biped_cfg=SceneEntityCfg(
            "biped",
            joint_names=BRAVER_biped_JPOINT_NAMES,
            preserve_order=True,
        ),
        quad_cfg=SceneEntityCfg(
            "quad",
            joint_names=BRAVER_QUAD_JOINT_NAMES,
            preserve_order=True,
        ),
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # active robot base measurements
        base_ang_vel = ObsTerm(
            func=mdp.active_base_ang_vel,
            params={
                "biped_cfg": SceneEntityCfg("biped"),
                "quad_cfg": SceneEntityCfg("quad"),
            },
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )

        proj_gravity = ObsTerm(
            func=mdp.active_projected_gravity,
            params={
                "biped_cfg": SceneEntityCfg("biped"),
                "quad_cfg": SceneEntityCfg("quad"),
            },
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # active robot joint measurements -> unified to 12
        joint_pos = ObsTerm(
            func=mdp.active_joint_pos_rel_12,
            params={
                "biped_cfg": SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES, preserve_order=True),
                "quad_cfg":  SceneEntityCfg("quad",  joint_names=BRAVER_QUAD_JOINT_NAMES,  preserve_order=True),
            },
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        joint_vel = ObsTerm(
            func=mdp.active_joint_vel_12,
            params={
                "biped_cfg": SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES, preserve_order=True),
                "quad_cfg":  SceneEntityCfg("quad",  joint_names=BRAVER_QUAD_JOINT_NAMES,  preserve_order=True),
            },
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        # last action (should already be 12-dim in your mixed action setup)
        last_action = ObsTerm(func=mdp.last_action)

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        robot_type = ObsTerm(func=mdp.robot_type_onehot)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        """History Observation for policy group"""

        base_ang_vel = ObsTerm(
            func=mdp.active_base_ang_vel,
            params={"biped_cfg": SceneEntityCfg("biped"), "quad_cfg": SceneEntityCfg("quad")},
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.active_projected_gravity,
            params={"biped_cfg": SceneEntityCfg("biped"), "quad_cfg": SceneEntityCfg("quad")},
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        joint_pos = ObsTerm(
            func=mdp.active_joint_pos_rel_12,
            params={
                "biped_cfg": SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES, preserve_order=True),
                "quad_cfg":  SceneEntityCfg("quad",  joint_names=BRAVER_QUAD_JOINT_NAMES,  preserve_order=True),
            },
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.active_joint_vel_12,
            params={
                "biped_cfg": SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES, preserve_order=True),
                "quad_cfg":  SceneEntityCfg("quad",  joint_names=BRAVER_QUAD_JOINT_NAMES,  preserve_order=True),
            },
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        # active robot base measurements
        base_lin_vel = ObsTerm(func=mdp.active_base_lin_vel, params={"biped_cfg": SceneEntityCfg("biped"), "quad_cfg": SceneEntityCfg("quad")}, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.active_base_ang_vel, params={"biped_cfg": SceneEntityCfg("biped"), "quad_cfg": SceneEntityCfg("quad")}, scale=0.25)
        proj_gravity = ObsTerm(func=mdp.active_projected_gravity, params={"biped_cfg": SceneEntityCfg("biped"), "quad_cfg": SceneEntityCfg("quad")})

        # active robot joint measurements -> 12
        joint_pos = ObsTerm(
            func=mdp.active_joint_pos_rel_12,
            params={
                "biped_cfg": SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES, preserve_order=True),
                "quad_cfg":  SceneEntityCfg("quad",  joint_names=BRAVER_QUAD_JOINT_NAMES,  preserve_order=True),
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.active_joint_vel_12,
            params={
                "biped_cfg": SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES, preserve_order=True),
                "quad_cfg":  SceneEntityCfg("quad",  joint_names=BRAVER_QUAD_JOINT_NAMES,  preserve_order=True),
            },
            scale=0.05,
        )

        last_action = ObsTerm(func=mdp.last_action)

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        robot_type = ObsTerm(func=mdp.robot_type_onehot)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_active_root = EventTerm(
        func=mdp.reset_root_state_uniform_multi,
        mode="reset",
        params={
            "biped_cfg": SceneEntityCfg("biped"),
            "quad_cfg": SceneEntityCfg("quad"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_active_joints = EventTerm(
        func=mdp.reset_joints_by_offset_set_multi,
        mode="reset",
        params={
            "biped_cfg": BRAVER_biped_PRESERVE_JOINT_ORDER_ASSET_CFG,
            "quad_cfg": BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG,
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    hide_inactive = EventTerm(
        func=mdp.hide_inactive_robot,
        mode="reset",
        params={
            "biped_cfg": SceneEntityCfg("biped"),
            "quad_cfg": SceneEntityCfg("quad"),
            "hidden_pos": (0.0, 0.0, -100.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
# class RewardsCfg:
#     """Minimal rewards for biped+quad (to make env runnable)."""
#     alive = RewTerm(
#         func=mdp.stay_alive_type_weighted,
#         weight=1.0,
#         params={
#             "w_biped": 6.0,
#             "w_quad": 0.0,
#         },
#     )
#     track_lin_vel_xy = RewTerm(
#         func=mdp.track_lin_vel_xy_exp_type_weighted,
#         weight=1.0,
#         params={
#             "command_name": "base_velocity",
#             "std_biped": 0.25,
#             "std_quad": 0.25,
#             "w_biped": 5.0,
#             "w_quad": 2.0,
#         },
#     )
#     track_ang_vel_z = RewTerm(
#         func=mdp.track_ang_vel_z_exp_type_weighted,
#         weight=1.0,
#         params={
#             "command_name": "base_velocity",
#             "std_biped": 0.25,
#             "std_quad": 0.25,
#             "w_biped": 2.5,
#             "w_quad": 1.0,
#         },
#     )
#     pen_lin_vel_z = RewTerm(
#         func=mdp.lin_vel_z_l2_type_weighted,
#         weight=1.0,
#         params={
#             "w_biped": -2.0,    
#             "w_quad": -2.0,
#         },
#     )  
#     pen_ang_vel_xy = RewTerm(
#         func=mdp.ang_vel_xy_l2_type_weighted,
#         weight=1.0,
#         params={
#             "w_biped": -0.5,    
#             "w_quad": -0.05,
#         },
#     ) 
#     action_rate = RewTerm(
#         func=mdp.action_rate_l2_type_weighted,
#         weight=1.0,
#         params={
#             "w_biped": -0.4 ,    
#             "w_quad": -0.02,
#         },
#     )
#     feet_air_time = RewTerm(
#         func=mdp.feet_air_time_type_weighted,
#         weight=1.0,
#         params=dict(
#             command_name="base_velocity",
#             threshold_biped=0.15,
#             threshold_quad=0.50,
#             w_biped=0.0,
#             w_quad=0.25,
#             biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
#             quad_sensor_cfg=SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
#         ),
#     ) 
#     undesired_contact = RewTerm(
#         func=mdp.undesired_contacts_type_weighted,
#         weight=1.0,
#         params={
#             "threshold": 1.0,
#             "w_biped": -10.0,    
#             "w_quad": -1.0,
#         },
#     )   
#     flat_orientation = RewTerm(
#         func=mdp.flat_orientation_l2_type_weighted,
#         weight=1.0,
#         params={
#             "w_biped":  -30.0,    
#             "w_quad":  -2.5,
#         },
#     )
#     joint_pos_limits = RewTerm(
#         func=mdp.joint_pos_limits_type_weighted,
#         weight=1.0,
#         params={
#             "w_biped": -0.3,    
#             "w_quad": -0.0,
#             "biped_cfg": BRAVER_biped_PRESERVE_JOINT_ORDER_ASSET_CFG,
#             "quad_cfg": BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG,
#         },
#     )

#     action_smoothness = RewTerm(
#         func=mdp.ActionSmoothnessPenalty_type,
#         weight=1.0,
#     )
#     base_height = RewTerm(
#         func=mdp.base_com_height_abs_type_weighted,
#         weight=1.0,
#         params={
#             "target_height_biped": 0.35,
#             "target_height_quad": 0.35,
#             "w_biped": -50.0,
#             "w_quad": 0.0,
#         },
#     )
#     feet_distance = RewTerm(
#         func=mdp.feet_distance_type_weighted,
#         weight=1.0,
#         params={
#             "w_biped": -30.0,    
#             "w_quad": -0.0,
#         },
#     ) 
#     feet_regulation = RewTerm(
#         func=mdp.feet_regulation_set_type_weighted,
#         weight=1.0,
#         params={
#             "foot_radius_biped": 0.03,
#             "foot_radius_quad": 0.03,
#             "base_height_target_biped": 0.35,
#             "base_height_target_quad": 0.35,
#             "w_biped": -1.0,    
#             "w_quad": -0.0,
#         },
#     )  
#     foot_landing_vel = RewTerm(
#         func=mdp.foot_landing_vel_type_weighted,
#         weight=1.0,
#         params={ 
#             "about_landing_threshold_biped": 0.05,
#             "about_landing_threshold_quad": 0.05,
#             "foot_radius_biped": 0.03,
#             "foot_radius_quad": 0.03,
#             "w_biped": -0.2,    
#             "w_quad": -0.0,
#         },
#     )    
#     feet_velocity = RewTerm(
#         func=mdp.feet_velocity_y_abs_sum_type_weighted,
#         weight=1.0,
#         params={ 
#             "w_biped": -0.8,    
#             "w_quad": -0.0,
#         },
#     )  
#     feet_clearance = RewTerm(
#         func=mdp.foot_clearance_reward1_type_weighted,  
#         weight=1.0,
#         params={ 
#             "target_height_biped": 0.10,
#             "target_height_quad": 0.10,
#             "std_biped": 0.05,
#             "std_quad": 0.25,
#             "tanh_mult_biped": 2.0,
#             "tanh_mult_quad": 2.0,
#             "w_biped": 2.0,   
#             "w_quad": 0.0,    
#         },
#     ) 
#     feet_gait = RewTerm(
#         func=mdp.feet_gait_type_weighted,  
#         weight=1.0,
#         params=dict(
#             # biped
#             period_biped=0.4,                 # 2Hz -> period=0.5s
#             offset_biped=[0.0, 0.5],
#             threshold_biped=0.5,
#             w_biped=5.0,
#             biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
           
#             # quad (示例 trot: FL & RR 同相，FR & RL 同相)
#             period_quad=0.4,
#             offset_quad=[0.0, 0.5, 0.5, 0.0], # FL,FR,RL,RR
#             threshold_quad=0.5,
#             w_quad=0.0,
#             quad_sensor_cfg=SceneEntityCfg("quad_contact_forces", body_names=BRAVER_QUAD_FOOT_NAMES),
#         ),
#     )   

#     # go1 rewardscfg 
#     track_default_joint_pos_exp = RewTerm(
#         func=mdp.track_default_joint_pos_exp_type_weighted,
#         weight=1.0,
#         params=dict(
#             std_biped=math.sqrt(0.15),
#             std_quad=math.sqrt(0.25),
#             w_biped=0.0,
#             w_quad=0.5,
#             biped_cfg=SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES),
#             quad_cfg=SceneEntityCfg("quad", joint_names=BRAVER_QUAD_JOINT_NAMES),
#         ),
#     )
#     feet_slide = RewTerm(
#         func=mdp.feet_slide_type_weighted,
#         weight=1.0,
#         params=dict(
#             w_biped=0.0,
#             w_quad=-0.2,
#             biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
#             quad_sensor_cfg=SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
#             biped_asset_cfg=SceneEntityCfg("biped", body_names=BRAVER_biped_FOOT_NAMES),
#             quad_asset_cfg=SceneEntityCfg("quad",  body_names=BRAVER_QUAD_FOOT_NAMES),
#             contact_force_threshold=1.0,
#         ),
#     )
#     feet_stumble = RewTerm(
#         func=mdp.feet_stumble_type_weighted,
#         weight=1.0,
#         params=dict(
#             biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
#             quad_sensor_cfg=SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
#             scale_biped=5.0,
#             scale_quad=5.0,
#             w_biped=0.0,
#             w_quad=-2.0,
#         ),
#     )

#     joint_deviation_hip = RewTerm(
#         func=mdp.joint_deviation_l1_type_weighted,
#         weight=1.0,
#         params=dict(
#             w_biped=0.0,
#             w_quad=-1.0,
#             biped_cfg=SceneEntityCfg("biped", joint_names=HIP_JOINT_NAMES),
#             quad_cfg=SceneEntityCfg("quad",  joint_names=".*_hip_joint"),
#         ),
#     )
#     joint_power_l2 = RewTerm(
#         func=mdp.joint_power_l2_type_weighted,
#         weight=1.0,
#         params=dict(
#             w_biped=0.0,
#             w_quad=-2.0e-4,
#             biped_cfg=BRAVER_biped_PRESERVE_JOINT_ORDER_ASSET_CFG,
#             quad_cfg=BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG,
#         ),
#     )
#     contact_force = RewTerm(
#         func=mdp.contact_forces_type_weighted,
#         weight=1.0,
#         params=dict(
#             threshold_biped=80.0,
#             threshold_quad=100.0,
#             w_biped=0.0,
#             w_quad=-0.001,
#             biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
#             quad_sensor_cfg=SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
#         ),
#     )
#     stand_still_without_cmd = RewTerm(
#         func=mdp.stand_still_joint_deviation_l1_type_weighted,
#         weight=1.0,
#         params=dict(
#             command_name="base_velocity",
#             command_threshold=0.06,
#             w_biped=0.0,
#             w_quad=-1.0,
#             biped_cfg=BRAVER_biped_PRESERVE_JOINT_ORDER_ASSET_CFG,
#             quad_cfg=BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG,
#         ),
#     )
#     trot_phase = RewTerm(
#         func=mdp.trot_typed_weight,
#         weight=1.0,
#         params=dict(
#             command_name="base_velocity",
#             w_biped=0.0,
#             w_quad=-0.1,
#             biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
#             quad_sensor_cfg=SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
#             # 这里的 joint_names 建议你只选“需要做对称约束的那几根关节”，保证顺序可控
#             biped_asset_cfg = BRAVER_biped_PRESERVE_JOINT_ORDER_ASSET_CFG,
#             quad_asset_cfg = BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG,
#         ),
#     )  

class RewardsCfg:
    """Minimal rewards for biped+quad (to make env runnable)."""
    alive = RewTerm(
        func=mdp.stay_alive_type_weighted,
        weight=1.0,
        params={
            "w_biped": 6.0,
            "w_quad": 10.0,
        },
    )
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp_type_weighted,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_biped": 0.2,
            "std_quad": 0.2,
            "w_biped": 5.0,
            "w_quad": 1.0,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_type_weighted,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_biped": 0.25,
            "std_quad": 0.25,
            "w_biped": 2.5,
            "w_quad": 0.5,
        },
    )
    base_height = RewTerm(
        func=mdp.base_com_height_abs_type_weighted,
        weight=1.0,
        params={
            "target_height_biped": 0.35,
            "target_height_quad": 0.35,
            "w_biped": -50.0,
            "w_quad": 0.0,
        },
    )
    pen_lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2_type_weighted,
        weight=1.0,
        params={
            "w_biped": -2.0,    
            "w_quad": -2.0,
        },
    )  
    pen_ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2_type_weighted,
        weight=1.0,
        params={
            "w_biped": -0.5,    
            "w_quad": -1.0,
        },
    ) 
    action_rate = RewTerm(
        func=mdp.action_rate_l2_type_weighted,
        weight=1.0,
        params={
            "w_biped": -0.4 ,    
            "w_quad": -0.01,
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_type_weighted,
        weight=1.0,
        params={
            "w_biped": -0.3,    
            "w_quad": -5.0,
            "biped_cfg": BRAVER_biped_PRESERVE_JOINT_ORDER_ASSET_CFG,
            "quad_cfg": BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG,
        },
    )
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts_type_weighted,
        weight=1.0,
        params={
            "threshold": 1.0,
            "w_biped": -10.0,    
            "w_quad": -1.0,
        },
    )
    action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty_type,
        weight=1.0,
    )
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2_type_weighted,
        weight=1.0,
        params={
            "w_biped":  -30.0,    
            "w_quad":  -50.0,
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_type_weighted,
        weight=1.0,
        params={
            "w_biped": -30.0,    
            "w_quad": -0.0,
        },
    ) 
    feet_regulation = RewTerm(
        func=mdp.feet_regulation_set_type_weighted,
        weight=1.0,
        params={
            "foot_radius_biped": 0.03,
            "foot_radius_quad": 0.03,
            "base_height_target_biped": 0.35,
            "base_height_target_quad": 0.35,
            "w_biped": -1.0,    
            "w_quad": -0.0,
        },
    )  
    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel_type_weighted,
        weight=1.0,
        params={ 
            "about_landing_threshold_biped": 0.05,
            "about_landing_threshold_quad": 0.05,
            "foot_radius_biped": 0.03,
            "foot_radius_quad": 0.03,
            "w_biped": -0.2,    
            "w_quad": -0.0,
        },
    )    
    feet_velocity = RewTerm(
        func=mdp.feet_velocity_y_abs_sum_type_weighted,
        weight=1.0,
        params={ 
            "w_biped": -0.8,    
            "w_quad": -0.0,
        },
    )  
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward1_type_weighted,  
        weight=1.0,
        params={ 
            "target_height_biped": 0.10,
            "target_height_quad": 0.10,
            "std_biped": 0.05,
            "std_quad": 0.25,
            "tanh_mult_biped": 2.0,
            "tanh_mult_quad": 2.0,
            "w_biped": 2.0,   
            "w_quad": 0.0,    
        },
    ) 
    feet_gait = RewTerm(
        func=mdp.feet_gait_type_weighted,  
        weight=1.0,
        params=dict(
            # biped
            period_biped=0.4,                 # 2Hz -> period=0.5s
            offset_biped=[0.0, 0.5],
            threshold_biped=0.5,
            w_biped=5.0,
            biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
           
            # quad (示例 trot: FL & RR 同相，FR & RL 同相)
            period_quad=0.4,
            offset_quad=[0.0, 0.5, 0.5, 0.0], # FL,FR,RL,RR
            threshold_quad=0.5,
            w_quad=0.0,
            quad_sensor_cfg=SceneEntityCfg("quad_contact_forces", body_names=BRAVER_QUAD_FOOT_NAMES),
        ),
    )   

    # go1 rewardscfg 
    # 1) joint default tracking（biped/quad 不同 std & 内部权重）
    track_default_joint_pos_exp = RewTerm(
        func=mdp.track_default_joint_pos_exp_type_weighted,
        weight=1.0,
        params=dict(
            std_biped=math.sqrt(0.15),
            std_quad=math.sqrt(0.25),
            w_biped=0.0,
            w_quad=0.25,
            biped_cfg=SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES),
            quad_cfg=SceneEntityCfg("quad", joint_names=BRAVER_QUAD_JOINT_NAMES),
        ),
    )

    # 2) feet slide（同一 term，不同内部权重）
    feet_slide = RewTerm(
        func=mdp.feet_slide_type_weighted,
        weight=1.0,
        params=dict(
            w_biped=0.0,
            w_quad=-0.1,
            biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
            quad_sensor_cfg=SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
            biped_asset_cfg=SceneEntityCfg("biped", body_names=BRAVER_biped_FOOT_NAMES),
            quad_asset_cfg=SceneEntityCfg("quad",  body_names=BRAVER_QUAD_FOOT_NAMES),
            contact_force_threshold=1.0,
        ),
    )

    # 3) hip joint deviation（biped/quad 权重不同）
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_type_weighted,
        weight=1.0,
        params=dict(
            w_biped=0.0,
            w_quad=-0.1,
            biped_cfg=SceneEntityCfg("biped", joint_names=HIP_JOINT_NAMES),
            quad_cfg=SceneEntityCfg("quad",  joint_names=".*_hip_joint"),
        ),
    )

    # 4) contact force penalty（阈值可不同）
    contact_force = RewTerm(
        func=mdp.contact_forces_type_weighted,
        weight=1.0,
        params=dict(
            threshold_biped=80.0,
            threshold_quad=100.0,
            w_biped=0.0,
            w_quad=-0.001,
            biped_sensor_cfg=SceneEntityCfg("biped_contact_forces", body_names=BRAVER_biped_FOOT_NAMES),
            quad_sensor_cfg=SceneEntityCfg("quad_contact_forces",  body_names=BRAVER_QUAD_FOOT_NAMES),
        ),
    )

    # 5) stand still when no cmd（内部权重可不同）
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_joint_deviation_l1_type_weighted,
        weight=1.0,
        params=dict(
            command_name="base_velocity",
            command_threshold=0.06,
            w_biped=0.0,
            w_quad=-1.0,
            biped_cfg=SceneEntityCfg("biped", joint_names=BRAVER_biped_JPOINT_NAMES),
            quad_cfg=SceneEntityCfg("quad",  joint_names=BRAVER_QUAD_JOINT_NAMES),
        ),
    )


@configclass
class TerminationsCfg:
    """Minimal terminations for biped+quad."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # fallen = DoneTerm(
    #     func=mdp.is_fallen,
    #     params={
    #         "biped_cfg": SceneEntityCfg("biped"),
    #         "quad_cfg": SceneEntityCfg("quad"),
    #         "min_height_biped": 0.18,
    #         "min_height_quad": 0.22,
    #         "grace_time": 0.2,
    #     },
    # )
    bad_contact = DoneTerm(
        func=mdp.illegal_contact_multi,
        params={
            "biped_sensor_cfg": SceneEntityCfg("biped_contact_forces",body_names=BRAVER_biped_BASE_NAME,),
            "quad_sensor_cfg": SceneEntityCfg("quad_contact_forces",body_names=BRAVER_QUAD_BASE_NAME,),
        },
    )
    # bad_posture = DoneTerm(
    #     func=mdp.bad_body_posture_multi,
    #     params={
    #         "biped_cfg": SceneEntityCfg("biped"),   
    #         "quad_cfg": SceneEntityCfg("quad"), 
    #     }
    # )
    
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_type_weighted)


##
# Environment configuration
##      

@configclass
class MultiLocoEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MultiLocoSceneCfg = MultiLocoSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()       
    # MDP settings
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    """Physics simulation configuration. Default is SimulationCfg()."""     
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 500
        self.sim.render_interval = self.decimation

        self.sim.physx.gpu_max_rigid_patch_count = 262144

@configclass
class MultiLocoRoughEnvCfg(MultiLocoEnvCfg):

    def __post_init__(self) -> None:

        super().__post_init__()
        #scene
        self.scene.terrain.max_init_terrain_level = 0
        #rewards
        # self.rewards.feet_air_time.params["w_quad"] = 0.01
        # self.rewards.flat_orientation.params["w_quad"] = 0.0
        #curriculum
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel_tracking_type_weighted

@configclass
class MultiLocoFlatEnvCfg(MultiLocoRoughEnvCfg):

    def __post_init__(self) -> None:

        super().__post_init__()
        #rewards
        # self.rewards.feet_air_time.params["w_quad"] = 0.25
        # self.rewards.flat_orientation.params["w_quad"] = -2.5
        #scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class MultiLocoRoughEnvCfg_Play(MultiLocoRoughEnvCfg):

    def __post_init__(self) -> None:

        super().__post_init__()
        #scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None

@configclass
class MultiLocoFlatEnvCfg_Play(MultiLocoFlatEnvCfg):

    def __post_init__(self) -> None:

        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

