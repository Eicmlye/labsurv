from configs.ocp._base_.params_benchmark import (
    AGENT_NUM,
    ALLOW_POLAR,
    BENCHMARK_NAME,
    CAM_TYPE_NUM,
    PAN_RANGE,
    PAN_SEC_NUM,
    TILT_RANGE,
    TILT_SEC_NUM,
)
from configs.runtime import DEVICE

env_cfg = dict(
    type="OCPMultiAgentPPOEnv",
    room_data_path=f"output/{BENCHMARK_NAME}/SurveillanceRoom.pkl",
    device=DEVICE,
    agent_num=AGENT_NUM,
    # visit_count_path="",
    pan_section_num=PAN_SEC_NUM,
    tilt_section_num=TILT_SEC_NUM,
    pan_range=PAN_RANGE,
    tilt_range=TILT_RANGE,
    allow_polar=ALLOW_POLAR,
    cam_types=CAM_TYPE_NUM,
    subgoals=[
        [0.1, 10],
        [0.2, 10],
        [0.3, 10],
        [0.4, 10],
        [0.5, 10],
        [0.55, 5],
        [0.6, 5],
        [0.65, 5],
        [0.7, 5],
        [0.75, 5],
        [0.8, 5],
        [0.85, 5],
        [0.9, 5],
        [0.95, 5],
    ],
    terminate_goal=1.0,
    reset_weight=4,
    individual_reward_alpha=0.8,  # must enable `mixed_reward` of agent
)
