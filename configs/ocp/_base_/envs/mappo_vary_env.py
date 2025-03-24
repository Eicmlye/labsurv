from configs.ocp._base_.params_vary import (
    AGENT_NUM,
    ALLOW_POLAR,
    CAM_TYPE_NUM,
    PAN_RANGE,
    PAN_SEC_NUM,
    ROOM_NAME,
    TILT_RANGE,
    TILT_SEC_NUM,
)
from configs.runtime import DEVICE

member_cfgs = [
    dict(
        type="OCPMultiAgentPPOEnv",
        room_data_path=f"output/{NAME}/SurveillanceRoom.pkl",
        device=DEVICE,
        agent_num=NUM,
        # visit_count_path=(
        #     "output/ocp/AGENT_NAME_benchmark/"
        #     "TASK_NAME/"
        #     "WORKING_DIR/"
        #     "envs/visit_count_episode_EPISODE.pkl"
        # ),
        pan_section_num=PAN_SEC_NUM,
        tilt_section_num=TILT_SEC_NUM,
        pan_range=PAN_RANGE,
        tilt_range=TILT_RANGE,
        allow_polar=ALLOW_POLAR,
        cam_types=TYPE_NUM,
        subgoals=(
            [[0.1 * i, 0.1] for i in range(1, 6)]  # 0.0 -> 0.5
            + [[0.5 + 0.05 * i, 0.05] for i in range(1, 7)]  # 0.5 -> 0.8
            + [[0.8 + 0.02 * i, 0.02] for i in range(1, 6)]  # 0.8 -> 0.9
            + [[0.9 + 0.01 * i, 0.01] for i in range(1, 6)]  # 0.9 -> 0.95
            + [[0.95 + 0.005 * i, 0.005] for i in range(1, 7)]  # 0.95 -> 0.98
            + [[0.98 + 0.001 * i, 0.001] for i in range(1, 20)]  # 0.98 -> 1.0
        ),
        terminate_goal=1.0,
        reset_weight=4,
        individual_reward_alpha=1.0,  # must enable `mixed_reward` of agent
    )
    for NAME, NUM, TYPE_NUM in zip(ROOM_NAME, AGENT_NUM, CAM_TYPE_NUM)
]

env_cfg = dict(
    type="OCPVaryEnv",
    member_cfgs=member_cfgs,
)
