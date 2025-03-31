from configs.ocp._base_.std_surveil_standard import cam_intrinsics as CAM_TYPES_STANDARD
from configs.ocp._base_.std_surveil_standard import (
    voxel_length as VOXEL_LENGTH_STANDARD,
)
from configs.ocp._base_.std_surveil_test import cam_intrinsics as CAM_TYPES_TEST
from configs.ocp._base_.std_surveil_test import voxel_length as VOXEL_LENGTH_TEST
from configs.ocp._base_.std_surveil_tiny import cam_intrinsics as CAM_TYPES_TINY
from configs.ocp._base_.std_surveil_tiny import voxel_length as VOXEL_LENGTH_TINY
from numpy import pi as PI

ROOM_NAME = "surv_room_test"
room_id = ROOM_NAME[10:]

AGENT_NUM = 10  # should be greater than 1
CAM_TYPE_NUM = len(
    CAM_TYPES_TEST
    if room_id == "test"
    else (CAM_TYPES_STANDARD if room_id == "standard" else CAM_TYPES_TINY)
)
PAN_SEC_NUM = 12
TILT_SEC_NUM = 9
PAN_RANGE = [-PI, PI]
TILT_RANGE = [-PI / 2, 0]
VOXEL_LENGTH = (
    VOXEL_LENGTH_TEST
    if room_id == "test"
    else (VOXEL_LENGTH_STANDARD if room_id == "standard" else VOXEL_LENGTH_TINY)
)
ALLOW_POLAR = True
MANUAL = False
BACKBONE_PATH = "labsurv/checkpoint/pointnet2backbone_rename.pth"
FREEZE_BACKBONE = [0, 1, 2, 3]
BACKBONE_RADIUS = 0.1
