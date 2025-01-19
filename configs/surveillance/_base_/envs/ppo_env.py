from configs.runtime import DEVICE

env_cfg = dict(
    type="OCPPPOEnv",
    room_data_path="output/surv_room/SurveillanceRoom.pkl",
    device=DEVICE,
    reward_goals=[
        0.2,
        0.5,
        0.7,
        0.8,
        0.9,
        0.95,
        1.0,
    ],  # 350
    reward_bonus=[
        [0.0, 0.05],  # 60
        [0.6, 0.02],  # 50
        [0.8, 0.005],  # 100
        [0.9, 0.001],  # 500
    ],
    terminate_goal=1.0,
)
