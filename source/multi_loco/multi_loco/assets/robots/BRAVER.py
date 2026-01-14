import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.actuators import DelayedPDActuatorCfg
from multi_loco.assets import BRAVERLAB_ASSET_DIR

# Urdf file cfg
BRAVER_biped_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{BRAVERLAB_ASSET_DIR}/resources/urdf/braver_biped.urdf",
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
            "L_abad_joint": 0.0,
            "R_abad_joint": 0.0,
            "L_hip_joint": -0.7,
            "R_hip_joint": -0.7,
            "L_knee_joint": 1.3,
            "R_knee_joint": 1.3,
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
                "L_abad_joint": 300.0,
                "L_hip_joint": 300.0,
                "L_knee_joint": 300.0,
                "R_abad_joint": 300.0,
                "R_hip_joint": 300.0,
                "R_knee_joint": 300.0,
            },
            velocity_limit={
                "L_abad_joint": 1000.0,
                "L_hip_joint": 1000.0,
                "L_knee_joint": 1000.0,
                "R_abad_joint": 1000.0,
                "R_hip_joint": 1000.0,
                "R_knee_joint": 1000.0,
            },
            stiffness={
                "L_abad_joint": 20.0,
                "L_hip_joint": 20.0,
                "L_knee_joint": 20.0,
                "R_abad_joint": 20.0,
                "R_hip_joint": 20.0,
                "R_knee_joint": 20.0,
            },
            damping={
                "L_abad_joint": 1.2,
                "L_hip_joint": 1.2,
                "L_knee_joint": 1.2,
                "R_abad_joint": 1.2,
                "R_hip_joint": 1.2,
                "R_knee_joint": 1.2,
            },
            armature={
                "L_abad_joint": 0.0,
                "L_hip_joint": 0.0,
                "L_knee_joint": 0.0,
                "R_abad_joint": 0.0,
                "R_hip_joint": 0.0,
                "R_knee_joint": 0.0,
            },
            friction=0,
            min_delay=0,
            max_delay=5,
            ),
    },
)

"""Configuration for the BRAVER robot."""


from isaaclab.managers import SceneEntityCfg


BRAVER_biped_PRESERVE_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
    "biped",
    joint_names=[
                "L_abad_joint","L_hip_joint","L_knee_joint",
                "R_abad_joint","R_hip_joint","R_knee_joint",],
    preserve_order=True,
)  # first left last right joint order


# link definitions
BRAVER_biped_BASE_NAME = "base_Link"
BRAVER_biped_FOOT_NAMES = ".*_foot"
BRAVER_biped_UNDESIRED_CONTACTS_NAMES = [BRAVER_biped_BASE_NAME,
    "L_Abad_Link",
    "L_hip_Link",
    "L_knee_Link",
    "R_Abad_Link",
    "R_hip_Link",
    "R_knee_Link",
    ]

BRAVER_biped_JPOINT_NAMES = [
                    "L_abad_joint","L_hip_joint","L_knee_joint",
                    "R_abad_joint","R_hip_joint","R_knee_joint",]   

BRAVER_biped_default_joint_pos = {
                "L_abad_joint": 0.0,
                "R_abad_joint": 0.0,        
                "L_hip_joint": -0.7,    
                "R_hip_joint": -0.7,    
                "L_knee_joint": 1.3,    
                "R_knee_joint": 1.3,    
            }

# joint definitions
HIP_JOINT_NAMES = ["L_abad_joint", "L_hip_joint", "R_abad_joint", "R_hip_joint"]
ZERO_JOINT_NAMES = ["L_abad_joint", "L_hip_joint", "R_abad_joint", "R_hip_joint"]

