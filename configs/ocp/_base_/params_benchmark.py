from configs.ocp._base_.std_surveil import cam_intrinsics as CAM_TYPES
from configs.ocp._base_.std_surveil import voxel_length
from numpy import pi as PI

BENCHMARK_NAME = "AC_03"
AGENT_NUM = 5  # should be greater than 1
CAM_TYPE_NUM = len(CAM_TYPES)
PAN_SEC_NUM = 8
TILT_SEC_NUM = 2
PAN_RANGE = [-PI, PI]
TILT_RANGE = [-PI / 2, 0]
VOXEL_LENGTH = voxel_length
ALLOW_POLAR = True
MANUAL = False
