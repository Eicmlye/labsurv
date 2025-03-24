from configs.ocp._base_.std_surveil import cam_intrinsics as CAM_TYPES_0
from configs.ocp._base_.std_surveil import voxel_length as VOXEL_LENGTH_0
from configs.ocp._base_.std_surveil_AC_01to09 import cam_intrinsics as CAM_TYPES_1
from configs.ocp._base_.std_surveil_AC_01to09 import voxel_length as VOXEL_LENGTH_1
from configs.ocp._base_.std_surveil_AC_10to32 import cam_intrinsics as CAM_TYPES_2
from configs.ocp._base_.std_surveil_AC_10to32 import voxel_length as VOXEL_LENGTH_2
from numpy import pi as PI

ROOM_NAME = ["AC_01", "AC_02", "AC_03", "AC_04", "AC_05", "AC_06"]
room_id = [
    int(name[-2:]) if name.startswith(("AC", "RW")) else None for name in ROOM_NAME
]

AGENT_NUM = [7, 4, 3, 5, 7, 10]  # should be greater than 1

assert len(ROOM_NAME) == len(AGENT_NUM)

CAM_TYPE_NUM = [
    len(CAM_TYPES_0 if id is None else (CAM_TYPES_1 if id < 10 else CAM_TYPES_2))
    for id in room_id
]
PAN_SEC_NUM = 8
TILT_SEC_NUM = 2
PAN_RANGE = [-PI, PI]
TILT_RANGE = [-PI / 2, 0]
VOXEL_LENGTH = [
    VOXEL_LENGTH_0 if id is None else (VOXEL_LENGTH_1 if id < 10 else VOXEL_LENGTH_2)
    for id in room_id
]
ALLOW_POLAR = True
MANUAL = False
BACKBONE_PATH = "labsurv/checkpoint/pointnet2backbone_melt.pth"
FREEZE_BACKBONE = []
BACKBONE_RADIUS = 0.1
