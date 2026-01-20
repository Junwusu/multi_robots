
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


UNITREE_GO1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.34),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "motor": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.7,
            velocity_limit=30,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            armature=1e-3,
            min_delay=0,
            max_delay=4,
        ),
    },
)

"""Configuration for the Unitree Go1 robot."""


from isaaclab.managers import SceneEntityCfg


# GO1_PRESERVE_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
#     "robot",
#     joint_names=[
#         "FL_hip_joint",
#         "FL_thigh_joint",
#         "FL_calf_joint",

#         "FR_hip_joint",
#         "FR_thigh_joint",
#         "FR_calf_joint",

#         "RL_hip_joint",
#         "RL_thigh_joint",
#         "RL_calf_joint",

#         "RR_hip_joint",
#         "RR_thigh_joint",
#         "RR_calf_joint",
#     ],
#     preserve_order=True,
# )

# GO1_BASE_NAME = "trunk"
# GO1_HIP_NAMES = ".*_hip"
# GO1_FOOT_NAMES = ".*_foot"
# GO1_UNDESIRED_CONTACTS_NAMES = [GO1_BASE_NAME, GO1_HIP_NAMES, GO1_HIP_NAMES]


BRAVER_QUAD_PRESERVE_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
    "quad",
    joint_names=[
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",

        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",

        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",

        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ],
    preserve_order=True,
)

BRAVER_QUAD_BASE_NAME = "trunk"
GO1_HIP_NAMES = ".*_hip"
BRAVER_QUAD_FOOT_NAMES = ".*_foot"
BRAVER_QUAD_UNDESIRED_CONTACTS_NAMES = [BRAVER_QUAD_BASE_NAME, ".*_hip", ".*_thigh", ".*_calf"]

BRAVER_QUAD_JOINT_NAMES = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",

        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",

        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",

        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]