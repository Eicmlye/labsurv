import pickle
from typing import Dict

from mmcv import Config

room_data_path = "output/surv_room/SurveillanceRoom.pkl"

with open(room_data_path, "rb") as f:
    df = pickle.load(f)

cam_cfg_path = df["cfg_path"]
ROOM_SHAPE = df["shape"]
INSTALL_PERMITTED = df["install_permitted"]

cfg = Config.fromfile(cam_cfg_path)
CAM_INTRINSICS: Dict = cfg.cam_intrinsics
