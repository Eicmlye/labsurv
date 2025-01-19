from configs.runtime import DEVICE

env_cfg = dict(
    type="OCPDDPGAddOnlyCleanEnv",
    room_data_path="output/surv_room/SurveillanceRoom.pkl",
    device=DEVICE,
)
