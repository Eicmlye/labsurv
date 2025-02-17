"""
Implementation and checkpoint from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Ref: https://github.com/charlesq34/pointnet2

Modified and documented by Eric
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Dropout,
    Module,
    ModuleList,
)


class PointNet2(Module):
    def __init__(self, num_classes):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = Conv1d(128, 128, 1)
        self.bn1 = BatchNorm1d(128)
        self.drop1 = Dropout(0.5)
        self.conv2 = Conv1d(128, num_classes, 1)

    def forward(self, data):
        """
        ## Arguments:

            data (Tensor): [B, N, DATA_DIM]
        """
        l0_data = data  # [B, N, DATA_DIM]
        l0_coords = data[:, :, :3]  # [B, N, 3]

        l1_coords, l1_data = self.sa1(l0_coords, l0_data)  # [B, 1024, 3], [B, 1024, 64]
        l2_coords, l2_data = self.sa2(l1_coords, l1_data)  # [B, 256, 3], [B, 256, 128]
        l3_coords, l3_data = self.sa3(l2_coords, l2_data)  # [B, 64, 3], [B, 64, 256]
        l4_coords, l4_data = self.sa4(l3_coords, l3_data)  # [B, 16, 3], [B, 16, 512]

        l3_data: Tensor = self.fp4(
            l3_coords, l4_coords, l3_data, l4_data
        )  # [B, 64, 256]
        l2_data: Tensor = self.fp3(
            l2_coords, l3_coords, l2_data, l3_data
        )  # [B, 256, 256]
        l1_data: Tensor = self.fp2(
            l1_coords, l2_coords, l1_data, l2_data
        )  # [B, 1024, 128]
        l0_data: Tensor = self.fp1(l0_coords, l1_coords, None, l1_data)  # [B, N, 128]

        l0_data = l0_data.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_data)), inplace=True))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x, l4_data


def squared_dist_mat(src: Tensor, dst: Tensor) -> Tensor:
    """
    ## Description:

        Compute L2 distances matrix of all pairs between `src` and `dst`.

        squared_dist{n,m} = (x_n - x_m)^2 + (y_n - y_m)^2 + (z_n - z_m)^2
                          = sum(src**2, dim=-1)
                            + sum(dst**2, dim=-1)
                            - 2 * src^T * dst

    ## Arguments:

        src (Tensor): [B, N, 3]

        dst (Tensor): [B, M, 3]

    ## Returns:

        squared_dist (Tensor): [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    squared_dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    squared_dist += torch.sum(src**2, -1).view(B, N, 1)
    squared_dist += torch.sum(dst**2, -1).view(B, 1, M)

    return squared_dist


def select_data(data: Tensor, index: Tensor):
    """
    ## Description:

        Pick out the data rows that `index` points to.

    ## Arguments:

        data (Tensor): [B, N, DATA_DIM]

        index (Tensor): [B, SAMPLE_NUM, ...], sample indices

    ## Returns:

        sample_data (Tensor): [B, SAMPLE_NUM, ..., DATA_DIM]

    ## Examples:
        >>> data  # [2, 4, 3]
        tensor([[[ 1,  2,  3],
                [ 4,  5,  6],
                [ 7,  8,  9],
                [10, 11, 12]],
                [[13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
                [22, 23, 24]]])
        >>> index  # [2, 2]
        tensor([[0, 1],
                [0, 2]])
        >>> sample_data  # [2, 2, 3]
        tensor([[[ 1,  2,  3],
                 [ 4,  5,  6]],
                [[13, 14, 15],
                 [19, 20, 21]]])
        >>> index  # [2, 2, 2]
        tensor([[[0, 1],
                 [0, 2]],
                [[2, 1],
                 [2, 0]]])
        >>> sample_data  # [2, 2, 2, 3]
        tensor([[[[ 1,  2,  3],
                  [ 4,  5,  6]],
                 [[ 1,  2,  3],
                  [ 7,  8,  9]]],
                [[[19, 20, 21],
                  [16, 17, 18]],
                 [[19, 20, 21],
                  [13, 14, 15]]]])
    """
    device = data.device
    batch_size = data.shape[0]

    view_shape = list(index.shape)  # == [B, SAMPLE_NUM, ...]
    view_shape[1:] = [1] * (len(view_shape) - 1)  # == [B, 1, 1...]
    repeat_shape = list(index.shape)  # == [B, SAMPLE_NUM, ...]
    repeat_shape[0] = 1  # == [1, SAMPLE_NUM, ...]
    batch_indices = (
        torch.arange(batch_size, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )

    sample_data = data[batch_indices, index, :]  # [B, SAMPLE_NUM, ..., DATA_DIM]

    return sample_data


def farthest_point_sampling(coords: Tensor, sample_num: int) -> Tensor:
    """
    ## Description:

        FPS algorithm.

    ## Arguments:

        coords (Tensor): [B, N, 3]

        sample_num (int): number of sample points.

    ## Returns:

        centroid_indices (Tensor): [B, SAMPLE_NUM], sample point indices.
    """
    device = coords.device
    batch_size, N, _ = coords.shape

    # [B, SAMPLE_NUM]
    centroid_indices = torch.zeros(batch_size, sample_num, dtype=torch.long).to(device)
    # [B, N]
    distance_to_nearest_centroid = torch.ones(batch_size, N).to(device) * float("inf")
    # randomly initialize a centroid index for every batch
    farthest = torch.randint(0, N, (batch_size,), dtype=torch.long).to(device)  # [B]
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)

    for i in range(sample_num):
        # choose the farthest points away from current centroids to be the new centroid
        centroid_indices[:, i] = farthest

        ## now update distance info
        # pick current centroid's coords out from every batch
        centroid_coords = coords[batch_indices, farthest, :].view(batch_size, 1, -1)
        # compute distance from centroid to every point
        distance_to_cur_centroid = torch.sum((coords - centroid_coords) ** 2, dim=-1)
        # update distance
        mask = distance_to_cur_centroid < distance_to_nearest_centroid
        distance_to_nearest_centroid[mask] = distance_to_cur_centroid[mask]

        ## the farthest point of every batch will be the new centroid
        farthest = torch.max(distance_to_nearest_centroid, dim=-1)[1]

    return centroid_indices  # [B, SAMPLE_NUM]


def group_points_to_samples(
    radius: float, max_group_member_num: int, coords: Tensor, sample_coords: Tensor
):
    """
    ## Arguments:

        radius (float): local region radius.

        max_group_member_num (int): max number of points in a group.

        coords (Tensor): [B, N, 3]

        sample_coords (Tensor): [B, SAMPLE_NUM, 3]

    ## Returns:

        group_index (Tensor): [B, SAMPLE_NUM, max_group_member_num]
    """
    device = coords.device
    batch_size, N, _ = coords.shape
    _, sample_num, _ = sample_coords.shape

    # init group indices
    grouped_index = (
        torch.arange(N, dtype=torch.long)
        .to(device)
        .view(1, 1, N)
        .repeat([batch_size, sample_num, 1])
    )
    # filter faraway points
    squared_dist = squared_dist_mat(sample_coords, coords)
    grouped_index[squared_dist > radius**2] = N
    # sort ascendingly to push non-member points to tail
    grouped_index = grouped_index.sort(dim=-1)[0][:, :, :max_group_member_num]
    # pick out the first point of every group
    group_first = (
        grouped_index[:, :, 0]
        .view(batch_size, sample_num, 1)
        .repeat([1, 1, max_group_member_num])
    )
    # pad non-member points with the first point if too few members are found
    mask = grouped_index == N
    grouped_index[mask] = group_first[mask]

    return grouped_index


def sample_and_group(
    sample_num: int,
    radius: float,
    max_group_member_num: int,
    coords: Tensor,
    data: Tensor,
    return_fps: bool = False,
) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    ## Arguments:

        sample_num (int)

        radius (float)

        max_group_member_num (int)

        coords (Tensor): [B, N, 3]

        data (Tensor): [B, N, DATA_DIM]

    ## Returns:

        sample_coords: [B, SAMPLE_NUM, max_group_member_num, 3]

        sample_coords_and_data: [B, SAMPLE_NUM, max_group_member_num, 3+DATA_DIM]
    """
    batch_size = coords.shape[0]
    fps_indices = farthest_point_sampling(coords, sample_num)  # [B, SAMPLE_NUM]
    sample_coords = select_data(coords, fps_indices)  # [B, SAMPLE_NUM, 3]
    # [B, SAMPLE_NUM, max_group_member_num, C]
    grouped_indices = group_points_to_samples(
        radius, max_group_member_num, coords, sample_coords
    )
    grouped_coords = select_data(coords, grouped_indices)  # [B, SAMPLE_NUM, 3]
    grouped_coords_under_sample_coord = grouped_coords - sample_coords.view(
        batch_size, sample_num, 1, -1
    )

    if data is not None:
        grouped_points = select_data(data, grouped_indices)
        sample_coords_and_data = torch.cat(
            [grouped_coords_under_sample_coord, grouped_points], dim=-1
        )  # [B, SAMPLE_NUM, max_group_member_num, 3+DATA_DIM]
    else:
        sample_coords_and_data = grouped_coords_under_sample_coord

    if return_fps:
        return sample_coords, sample_coords_and_data, grouped_coords, fps_indices
    else:
        return sample_coords, sample_coords_and_data


def sample_and_group_all(coords: Tensor, data: Tensor):
    """
    ## Arguments:

        coords (Tensor): [B, N, 3]

        data (Tensor): [B, N, DATA_DIM]

    ## Returns:

        sample_coords: [B, 1, 3]

        sample_coords_and_data: [B, 1, N, 3+DATA_DIM]
    """
    device = coords.device
    B, N, C = coords.shape

    sample_coords = torch.zeros(B, 1, C).to(device)
    grouped_coords = coords.view(B, 1, N, C)
    if data is not None:
        sample_coords_and_data = torch.cat(
            [grouped_coords, data.view(B, 1, N, -1)], dim=-1
        )
    else:
        sample_coords_and_data = grouped_coords

    return sample_coords, sample_coords_and_data


class PointNetSetAbstraction(Module):
    def __init__(
        self,
        sample_num: int,
        radius: float,
        max_group_member_num: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool,
    ):
        super().__init__()
        self.sample_num = sample_num
        self.radius = radius
        self.max_group_member_num = max_group_member_num

        self.mlp_convs = ModuleList()
        self.mlp_bns = ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(BatchNorm2d(out_channel))
            last_channel = out_channel

        self.group_all = group_all

    def forward(self, coords: Tensor, data: Tensor):
        """
        ## Arguments:

            coords (Tensor): [B, N, 3]

            data (Tensor): [B, N, DATA_DIM]

        ## Returns:

            sample_coords (Tensor): [B, SAMPLE_NUM, 3]

            sample_feats (Tensor): [B, SAMPLE_NUM, FEAT_DIM]
        """

        if self.group_all:
            sample_coords, sample_coords_and_data = sample_and_group_all(coords, data)
        else:
            sample_coords, sample_coords_and_data = sample_and_group(
                self.sample_num, self.radius, self.max_group_member_num, coords, data
            )

        sample_feats = sample_coords_and_data.permute(
            0, 3, 2, 1
        )  # [B, 3+DATA_DIM, max_group_member_num, SAMPLE_NUM]
        for i, conv in enumerate(self.mlp_convs):
            sample_feats = F.relu(self.mlp_bns[i](conv(sample_feats)))

        # [B, SAMPLE_NUM, FEAT_DIM]
        sample_feats = torch.max(sample_feats, 2)[0].permute(0, 2, 1)

        return sample_coords, sample_feats


class PointNetFeaturePropagation(Module):
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()

        self.mlp_convs = ModuleList()
        self.mlp_bns = ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self, coords: Tensor, sample_coords: Tensor, data: Tensor, sample_data: Tensor
    ):
        """
        ## Arguments:

            coords (Tensor): [B, N, 3]

            sample_coords (Tensor): [B, SAMPLE_NUM, 3]

            data (Tensor): [B, N, DATA_DIM]

            sample_data (Tensor): [B, SAMPLE_NUM, DATA_DIM]

        ## Returns:

            new_points (Tensor): [B, N, FEAT_DIM]
        """
        batch_size, N, _ = coords.shape
        _, sample_num, _ = sample_coords.shape

        if sample_num == 1:
            interpolated_data = sample_data.repeat(1, N, 1)  # [B, N, 1]
        else:
            dists = squared_dist_mat(coords, sample_coords)  # [B, N, SAMPLE_NUM]
            dists, idx = dists.sort(dim=-1)  # [B, N, SAMPLE_NUM], [B, N, SAMPLE_NUM]
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3], [B, N, 3]

            dist_reciprocals = 1.0 / (dists + 1e-8)  # [B, N, 3]
            norm = torch.sum(dist_reciprocals, dim=2, keepdim=True)
            weight = dist_reciprocals / norm  # [B, N, 3]
            interpolated_data = torch.sum(  # [B, N, 1]
                # [B, N, 3, 3] * [B, N, 3, 1]
                select_data(sample_coords, idx) * weight.view(batch_size, N, 3, 1),
                dim=2,
            )

        if data is not None:
            # [B, N, 1+DATA_DIM]
            upsampled_data = torch.cat([data, interpolated_data], dim=-1)
        else:
            # [B, N, 1]
            upsampled_data = interpolated_data

        upsampled_data = upsampled_data.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            upsampled_data = F.relu(bn(conv(upsampled_data)))

        return upsampled_data.permute(0, 2, 1)  # [B, N, FEAT_DIM]
