from configs.runtime import DEVICE
from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES
from numpy import pi as PI

pan_section_num = 12
tilt_section_num = 9
cam_type_num = len(CAM_TYPES)

env_cfg = dict(
    type="OCPMultiAgentPPOEnv",
    room_data_path="output/surv_room_test/SurveillanceRoom.pkl",
    device=DEVICE,
    agent_num=1,
    reset_count_path="output/ocp/test_mappo/std_realtime_3w/envs/visit_count_episode_10000.pkl",
    pan_section_num=pan_section_num,
    tilt_section_num=tilt_section_num,
    pan_range=[-PI, PI],
    tilt_range=[-PI / 2, 0],
    cam_types=cam_type_num,
    terminate_goal=0.15,
    # vismask_path="output/ocp/test_mappo/std_speedup/envs/vismasks.pkl",
    # cache_vismask=False,
)
