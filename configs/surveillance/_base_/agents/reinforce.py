import math

from configs.runtime import DEVICE
from configs.surveillance._base_.envs import (
    CAM_INTRINSICS,
    INSTALL_PERMITTED,
    ROOM_SHAPE,
)

agent_cfg = dict(
    episode_based=True,
    agent=dict(
        type="REINFORCE",
        device=DEVICE,
        gamma=0.98,
        policy_net_cfg=dict(
            type="SimplePolicyNet",
            device=DEVICE,
            state_dim=ROOM_SHAPE[0] * ROOM_SHAPE[1] * ROOM_SHAPE[2] * 7,
            hidden_dim=256,
            action_dim=2,
            # [is_discrete, lbound, ubound(, include_lbound, include_ubound)]
            extra_params=[
                [True, 0, len(INSTALL_PERMITTED) - 1],  # pos
                [False, -math.pi, math.pi, True, False],  # pan
                [False, -math.pi / 2, math.pi / 2, True, True],  # tilt
                [True, 0, len(CAM_INTRINSICS) - 1],  # cam_type
            ],
        ),
        lr=5e-5,
        # resume_from=None,
        # load_from=None,
    ),
)
