import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.actuators import DelayedPDActuatorCfg
from multi_loco.assets import BRAVERLAB_ASSET_DIR

# Urdf file cfg
BRAVER_quad_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{BRAVERLAB_ASSET_DIR}/resources/urdf/braver_quad.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
       
    ),
    debug_vis=True, # Enable debug visualization
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            "FL_abad_joint": 0.0,
            "FR_abad_joint": 0.0,
            "FL_hip_joint": -0.7,
            "FR_hip_joint": -0.7,
            "FL_knee_joint": 1.3,
            "FR_knee_joint": 1.3,
            "RL_abad_joint": 0.0,
            "RR_abad_joint": 0.0,
            "RL_hip_joint": -0.7,
            "RR_hip_joint": -0.7,
            "RL_knee_joint": 1.3,
            "RR_knee_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "motor": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_joint",
            ],
            effort_limit={
                "FL_abad_joint": 300.0,
                "FL_hip_joint": 300.0,
                "FL_knee_joint": 300.0,
                "FR_abad_joint": 300.0,
                "FR_hip_joint": 300.0,
                "FR_knee_joint": 300.0,
                "RL_abad_joint": 300.0,
                "RL_hip_joint": 300.0,
                "RL_knee_joint": 300.0,
                "RR_abad_joint": 300.0,
                "RR_hip_joint": 300.0,
                "RR_knee_joint": 300.0,
            },
            velocity_limit={
                "FL_abad_joint": 1000.0,
                "FL_hip_joint": 1000.0,
                "FL_knee_joint": 1000.0,
                "FR_abad_joint": 1000.0,
                "FR_hip_joint": 1000.0,
                "FR_knee_joint": 1000.0,
                "RL_abad_joint": 1000.0,
                "RL_hip_joint": 1000.0,
                "RL_knee_joint": 1000.0,
                "RR_abad_joint": 1000.0,
                "RR_hip_joint": 1000.0,
                "RR_knee_joint": 1000.0,
            },
            stiffness={
                "FL_abad_joint": 20.0,
                "FL_hip_joint": 20.0,
                "FL_knee_joint": 20.0,
                "FR_abad_joint": 20.0,
                "FR_hip_joint": 20.0,
                "FR_knee_joint": 20.0,
                "RL_abad_joint": 20.0,
                "RL_hip_joint": 20.0,
                "RL_knee_joint": 20.0,
                "RR_abad_joint": 20.0,
                "RR_hip_joint": 20.0,
                "RR_knee_joint": 20.0,
            },
            damping={
                "FL_abad_joint": 1.2,
                "FL_hip_joint": 1.2,
                "FL_knee_joint": 1.2,
                "FR_abad_joint": 1.2,
                "FR_hip_joint": 1.2,
                "FR_knee_joint": 1.2,
                "RL_abad_joint": 1.2,
                "RL_hip_joint": 1.2,
                "RL_knee_joint": 1.2,
                "RR_abad_joint": 1.2,
                "RR_hip_joint": 1.2,
                "RR_knee_joint": 1.2,
            },
            armature={
                "FL_abad_joint": 0.0,
                "FL_hip_joint": 0.0,
                "FL_knee_joint": 0.0,
                "FR_abad_joint": 0.0,
                "FR_hip_joint": 0.0,
                "FR_knee_joint": 0.0,
                "RL_abad_joint": 0.0,
                "RL_hip_joint": 0.0,
                "RL_knee_joint": 0.0,
                "RR_abad_joint": 0.0,
                "RR_hip_joint": 0.0,
                "RR_knee_joint": 0.0,
            },
            friction=0,
            min_delay=0,
            max_delay=5,
            ),
    },
)

"""Configuration for the BRAVER robot."""


from isaaclab.managers import SceneEntityCfg


BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        'FL_abad_joint', 
        'FL_hip_joint', 
        'FL_knee_joint', 
        'FR_abad_joint', 
        'FR_hip_joint', 
        'FR_knee_joint', 
        'RL_abad_joint', 
        'RL_hip_joint', 
        'RL_knee_joint', 
        'RR_abad_joint', 
        'RR_hip_joint', 
        'RR_knee_joint', 
    ],
    preserve_order=True,
)  # first left last right joint order


# link definitions
BRAVER_QUAD_BASE_NAME = "base",
BRAVER_QUAD_FOOT_NAMES = ["FL_foot", "FR_foot","RL_foot", "RR_foot"]
BRAVER_QUAD_UNDESIRED_CONTACTS_NAMES = ["base",
    "FL_Abad_Link",
    "FL_hip_Link",
    "FL_knee_Link",
    "FR_Abad_Link",
    "FR_hip_Link",
    "FR_knee_Link",
    "RL_Abad_Link",
    "RL_hip_Link",
    "RL_knee_Link",
    "RR_Abad_Link",
    "RR_hip_Link",
    "RR_knee_Link",
    ]

BRAVER_QUAD_JOINT_NAMES = [
                    "FL_abad_joint","FL_hip_joint","FL_knee_joint",
                    "FR_abad_joint","FR_hip_joint","FR_knee_joint",
                    "RL_abad_joint","RL_hip_joint","RL_knee_joint",
                    "RR_abad_joint","RR_hip_joint","RR_knee_joint",]

BRAVER_QUAD_default_joint_pos = {
                "FL_abad_joint": 0.0,
                "FR_abad_joint": 0.0,
                "RL_abad_joint": 0.0,
                "RR_abad_joint": 0.0,
                "FL_hip_joint": -0.7,
                "FR_hip_joint": -0.7,
                "RL_hip_joint": -0.7,
                "RR_hip_joint": -0.7,
                "FL_knee_joint": 1.3,
                "FR_knee_joint": 1.3,
                "RL_knee_joint": 1.3,
                "RR_knee_joint": 1.3,
            }
# joint definitions
BRAVER_QUAD_HIP_JOINT_NAMES = ["L_abad_joint", "L_hip_joint", "R_abad_joint", "R_hip_joint"]
BRAVER_QUAD_ZERO_JOINT_NAMES = ["L_abad_joint", "L_hip_joint", "R_abad_joint", "R_hip_joint"]

