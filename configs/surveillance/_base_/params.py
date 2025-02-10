from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES
from numpy import pi as PI

AGENT_NUM = 2  # should be greater than 1
CAM_TYPE_NUM = len(CAM_TYPES)
PAN_SEC_NUM = 12
TILT_SEC_NUM = 9
PAN_RANGE = [-PI, PI]
TILT_RANGE = [-PI / 2, 0]
