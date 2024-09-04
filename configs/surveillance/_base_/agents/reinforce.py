from configs.runtime import DEVICE

agent_cfg = dict(
    episode_based=True,
    agent=dict(
        type="OCPREINFORCE",
        device=DEVICE,
        gamma=0.98,
        policy_net_cfg=dict(
            type="OCPREINFORCEPolicyNet",
            device=DEVICE,
            state_dim=12,
            hidden_dim=256,
            action_dim=4,  # add, del, adjust, stop
            params_dim=4,  # pos_index, pan, tilt, cam_type
        ),
        lr=5e-5,
        # resume_from=None,
        # load_from="output/surveillance/episode_6500.pth",
    ),
)
