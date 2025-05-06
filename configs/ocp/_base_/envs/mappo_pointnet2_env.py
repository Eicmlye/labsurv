from configs.ocp._base_.params import (
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

env_cfg = dict(
    type="OCPMultiAgentPPOEnv",
    room_data_path=f"output/{ROOM_NAME}/SurveillanceRoom.pkl",
    device=DEVICE,
    agent_num=AGENT_NUM,
    # visit_count_path=(
    #     "output/ocp/AGENT_NAME/"
    #     "TASK_NAME/"
    #     "WORKING_DIR/"
    #     "envs/visit_count_episode_EPISODE.pkl"
    # ),
    pan_section_num=PAN_SEC_NUM,
    tilt_section_num=TILT_SEC_NUM,
    pan_range=PAN_RANGE,
    tilt_range=TILT_RANGE,
    allow_polar=ALLOW_POLAR,
    cam_types=CAM_TYPE_NUM,
    reset_rand_prob=0,
    reset_pos="center",
    subgoals=(
        [
            0.4,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.825,
            0.85,
            0.875,
            0.9,
            0.925,
            0.95,
            0.96,
            0.97,
            0.98,
            0.99,
            1.0,
        ]
        # [[0.1 * i, 0.1] for i in range(1, 6)]  # 0.0 -> 0.5
        # + [[0.5 + 0.05 * i, 0.05] for i in range(1, 7)]  # 0.5 -> 0.8
        # + [[0.8 + 0.02 * i, 0.02] for i in range(1, 6)]  # 0.8 -> 0.9
        # + [[0.9 + 0.01 * i, 0.01] for i in range(1, 6)]  # 0.9 -> 0.95
        # + [[0.95 + 0.005 * i, 0.005] for i in range(1, 7)]  # 0.95 -> 0.98
        # + [[0.98 + 0.001 * i, 0.001] for i in range(1, 20)]  # 0.98 -> 1.0
        # + [  # inv rew 0.2 -> 0.3, total 0.05
        #     [0.2 + 0.01 * i, 0.001 * 0.5] for i in range(1, 11)
        # ]
        # + [  # inv rew 0.3 -> 0.4, total 0.1
        #     [0.3 + 0.01 * i, 0.002 * 0.5] for i in range(1, 11)
        # ]
        # + [  # inv rew 0.4 -> 0.5, total 0.15
        #     [0.4 + 0.01 * i, 0.003 * 0.5] for i in range(1, 11)
        # ]
        # + [  # inv rew 0.5 -> 0.6, total 0.2
        #     [0.5 + 0.01 * i, 0.004 * 0.5] for i in range(1, 11)
        # ]
        # + [  # inv rew 0.6 -> 0.7, total 0.05
        #     [0.6 + 0.01 * i, 0.001 * 0.5] for i in range(1, 11)
        # ]
        # + [  # inv rew 0.7 -> 0.8, total 0.1
        #     [0.7 + 0.01 * i, 0.002 * 0.5] for i in range(1, 11)
        # ]
        # + [  # inv rew 0.8 -> 0.9, total 0.15
        #     [0.8 + 0.01 * i, 0.003 * 0.5] for i in range(1, 11)
        # ]
        # + [  # inv rew 0.9 -> 1.0, total 0.2
        #     [0.9 + 0.01 * i, 0.004 * 0.5] for i in range(1, 10)
        # ]
    ),
    terminate_goal=0.65,
    reset_weight=4,
    individual_reward_alpha=0.1,  # must enable `mixed_reward` of agent
)
