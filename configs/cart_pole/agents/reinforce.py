from configs.runtime import DEVICE

agent_cfg = dict(
    episode_based = True,

    agent = dict(
        type="REINFORCE",
        device=DEVICE,
        gamma=0.98,
        policy_net_cfg=dict(
            type="SimplePolicyNet",
            state_dim=4,
            hidden_dim=256,
            action_dim=2,
        ),
        lr=5e-5,
        # resume_from="output/cart_pole_reinforce/episode_10000.pth",
        # load_from="output/cart_pole_reinforce/episode_50000.pth",
    ),
)
