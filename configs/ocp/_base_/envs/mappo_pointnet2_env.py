from configs.ocp._base_.params import (
    AGENT_NUM,
    CAM_TYPE_NUM,
    PAN_RANGE,
    PAN_SEC_NUM,
    TILT_RANGE,
    TILT_SEC_NUM,
)
from configs.runtime import DEVICE

env_cfg = dict(
    type="OCPMultiAgentPPOEnv",
    room_data_path="output/surv_room_test/SurveillanceRoom.pkl",
    device=DEVICE,
    agent_num=AGENT_NUM,
    # visit_count_path="output/ocp/AGENT_NAME/EXP_NAME/envs/visit_count_episode_EPI_NUM.pkl",
    pan_section_num=PAN_SEC_NUM,
    tilt_section_num=TILT_SEC_NUM,
    pan_range=PAN_RANGE,
    tilt_range=TILT_RANGE,
    cam_types=CAM_TYPE_NUM,
    subgoals=[
        [0.5, 10],
        [0.6, 10],
        [0.65, 10],
        [0.7, 10],
        [0.75, 10],
        [0.8, 20],
        [0.825, 20],
        [0.85, 20],
        [0.875, 20],
    ],
    terminate_goal=0.9,
    reset_weight=4,
    # individual_reward_alpha=0.8,  # must enable `mixed_reward` of agent
)
