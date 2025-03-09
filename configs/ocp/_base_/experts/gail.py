from configs.runtime import DEVICE
from configs.ocp._base_.params_benchmark import (
    AGENT_NUM,
    CAM_TYPE_NUM,
    VOXEL_LENGTH,
)

expert = dict(
    type="GAIL",
    disctiminator_cfg=dict(
        type="PointNet2Discriminator",
        device=DEVICE,
        hidden_dim=16,
        cam_types=CAM_TYPE_NUM,
        comm_attn_head_num=4,
        neigh_out_dim=32,
        voxel_length=VOXEL_LENGTH,
    ),
    device=DEVICE,
    lr=1e-5,
    agent_num=AGENT_NUM,
    sample_batchsize=15,
    gradient_accumulation_batchsize=20,
    backbone_path="labsurv/checkpoint/pointnet2backbone_rename.pth",
    freeze_backbone=[0, 1, 2, 3],
    expert_data_path="output/expert",
)