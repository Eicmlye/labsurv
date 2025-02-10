from configs.runtime import DEVICE
from configs.surveillance._base_.params import (
    AGENT_NUM,
    CAM_TYPE_NUM,
    PAN_RANGE,
    PAN_SEC_NUM,
    TILT_RANGE,
    TILT_SEC_NUM,
)

env_cfg = dict(
    type="OCPMultiAgentPPOEnv",
    room_data_path="output/surv_room_test/SurveillanceRoom.pkl",
    device=DEVICE,
    agent_num=AGENT_NUM,
    visit_count_path="output/ocp/test_mappo/ma6_realtime_70from50_rewarddelta/envs/visit_count_episode_400.pkl",
    pan_section_num=PAN_SEC_NUM,
    tilt_section_num=TILT_SEC_NUM,
    pan_range=PAN_RANGE,
    tilt_range=TILT_RANGE,
    cam_types=CAM_TYPE_NUM,
    subgoals=[
        [0.5, 20],
        [0.6, 20],
        [0.65, 20],
    ],
    terminate_goal=0.7,
    reset_weight=4,
    # individual_reward_alpha=0.8,  # must enable `mixed_reward` of agent
    # vismask_path="output/ocp/test_mappo/std_speedup/envs/vismasks.pkl",
    # cache_vismask=False,
)
