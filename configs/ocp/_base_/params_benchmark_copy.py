from configs.ocp._base_.std_surveil_AC_01to09 import cam_intrinsics as CAM_TYPES_1
from configs.ocp._base_.std_surveil_AC_01to09 import voxel_length as VOXEL_LENGTH_1
from configs.ocp._base_.std_surveil_AC_10to32 import cam_intrinsics as CAM_TYPES_2
from configs.ocp._base_.std_surveil_AC_10to32 import voxel_length as VOXEL_LENGTH_2
from numpy import pi as PI

BENCHMARK_NAME = "AC_04"
benchmark_id = int(BENCHMARK_NAME[-2:])

AGENT_NUM = 5  # should be greater than 1
CAM_TYPE_NUM = len(CAM_TYPES_1 if benchmark_id < 10 else CAM_TYPES_2)
PAN_SEC_NUM = 8
TILT_SEC_NUM = 2
PAN_RANGE = [-PI, PI]
TILT_RANGE = [-PI / 2, 0]
VOXEL_LENGTH = VOXEL_LENGTH_1 if benchmark_id < 10 else VOXEL_LENGTH_2
ALLOW_POLAR = True
MANUAL = False
BACKBONE_PATH = "labsurv/checkpoint/pointnet2backbone_rename.pth"
FREEZE_BACKBONE = [0, 1]
BACKBONE_RADIUS = 0.1
