from configs.ocp._base_.std_surveil import cam_intrinsics as CAM_TYPES
from configs.ocp._base_.std_surveil import voxel_length
from numpy import pi as PI

AGENT_NUM = 10  # should be greater than 1
CAM_TYPE_NUM = len(CAM_TYPES)
PAN_SEC_NUM = 12
TILT_SEC_NUM = 9
PAN_RANGE = [-PI, PI]
TILT_RANGE = [-PI / 2, 0]
VOXEL_LENGTH = voxel_length
ALLOW_POLAR = False
