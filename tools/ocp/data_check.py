import argparse
from typing import List, Optional, Tuple
import os.path as osp

import numpy as np
import torch
from configs.runtime import DEVICE as DEVICE_STR
from labsurv.physics import SurveillanceRoom
from torch import pi as PI
from torch import Tensor

from labsurv.utils.surveillance import (
    concat_points_with_color,
    save_visualized_points,
)

DEVICE = torch.device(DEVICE_STR)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark room builder.")

    parser.add_argument(  # --sample-name, -n
        "--sample-name", "-n", type=str, help="Name of the sample."
    )
    parser.add_argument(  # --data-path, -d
        "--data-path", "-d", type=str, help="Directory path of data files."
    )
    parser.add_argument(  # --cam
        "--cam",
        type=float,
        nargs="+",
        help="Positional and directional indices of the camera."
    )

    args = parser.parse_args()

    return args


def _get_room_params(spec_path: str, sample_name: str) -> Tuple[List[int], int]:
    """
    ## Arguments:

        spec_path (str): path of "XX_specs.txt".

        sample_name (str): should be in the format of AC_XX or RW_XX.

    ## Returns:

        room_shape (List[int]): the W, D, H of the room.

        room_h (int): the height without camera permitted area.
    """

    params: str = None
    with open(spec_path, "r") as f:
        for line in f:
            cur_params = line.strip().split()
            if cur_params[0] == sample_name:
                params = cur_params
                break

    (
        sample_max_w,
        sample_max_d,
        sample_max_h,
        cam_h,
        _,
        _,
        _,
        _,
        voxel_length,
        _,
    ) = [float(param) for param in params[1:]]

    room_shape = [
        int(sample_max_w / voxel_length) + 1,
        int(sample_max_d / voxel_length) + 1,
        int(cam_h / voxel_length) + 1,
    ]

    return room_shape, int(sample_max_h / voxel_length) + 1


def _cam_params2index(room_shape: List[int], cam_params: List[int]):
    pan_index: int = -1
    tilt_index: int = -1

    W, D, _ = room_shape
    benchmark_pan = (
        cam_params[3]
        if cam_params[3] <= PI and cam_params[3] >= 0
        else 2 * PI + cam_params
    )
    pan_index = int(round(benchmark_pan / (PI / 4)))
    tilt_index = int(round((0 - cam_params[4]) / (PI / 4)))

    cam_index = (
        pan_index
        + tilt_index * 8
        + int(cam_params[1]) * 8 * 3
        + int(cam_params[0]) * 8 * 3 * D
    )

    return cam_index


def _get_cam_covering_samples(
    cover_path: str, room_shape: List[int], cam_params: List[int]
) -> Tuple[List[int], int]:
    """
    ## Arguments:

        cover_path (str): path of "XX_xx_cover.txt".

        room_shape (List[int]): W, D, H of the room.

        cam_params (List[int]): x, y, z, p, t of the camera.

    ## Returns:

        camera_covering_samples (List[int]): a list of sample indices covered by the
        camera.
    """
    
    sample_num: Optional[int] = None
    candidate_num: Optional[int] = None

    with open(cover_path, "r") as f:
        count_line = -1
        sample_index = None
        covering_candidate_num = None
        covering_candidates = []
        cam_index = _cam_params2index(room_shape, cam_params)
        camera_covering_samples: List[int] = []

        for line in f:
            if sample_num is None:
                sample_num, candidate_num = list(map(int, line.strip().split()))
            elif count_line % 3 == 0:
                sample_index = int(line.strip())
            elif count_line % 3 == 1:
                covering_candidate_num = int(line.strip())
            elif count_line % 3 == 2:
                covering_candidates = list(map(int, line.strip().split()))

            if (
                count_line >= 0
                and count_line % 3 == 2
                and cam_index in covering_candidates
            ):
                camera_covering_samples.append(sample_index)

            count_line += 1

    return camera_covering_samples, cam_index


def _sample_index2coords(room_shape: List[int], room_h: int, sample_indices: List[int]):
    """
    ## Arguments:

        room_shape (List[int])

        sample_indices (List[int])
    
    ## Returns:

        sample_coords (Tensor): [N, 3]
    """

    sample_coords = []
    W, D, _ = room_shape
    H = room_h
    for sample_index in sample_indices:
        sample_coord = [-1, -1, -1]

        sample_coord[0] = sample_index // (D * H)
        sample_coord[1] = sample_index % (D * H) // H
        sample_coord[2] = sample_index % H

        sample_coords.append(sample_coord)

    return torch.tensor(sample_coords, dtype=torch.int64, device=DEVICE)


def _build_room(
    room_shape: List[int], room_h: int, cfg_path: str, cam_params: List[int]
) -> SurveillanceRoom:
    room = SurveillanceRoom(
        device=DEVICE,
        cfg_path=cfg_path,
        shape=room_shape,
    )
    print("Building occupancy blocks...")

    print("Building permission blocks...")
    room.add_block(
        [room_shape[0], room_shape[1], 1],
        point_type="install_permitted",
        displacement=np.array([0, 0, room_h]),
    )

    print("Building monitored blocks...")
    resol_requirement = dict(
        h_res_req_min=500,
        h_res_req_max=1000,
        v_res_req_min=500,
        v_res_req_max=1000,
    )
    room.add_block(
        [room_shape[0], room_shape[1], room_h],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    
    print("Adding cameras...")
    room.add_cam(
        cam_params[:3],
        cam_params[3:],
        "std_cam",
    )

    return room


def main(sample_name: str, data_path: str, cam_params: List[int]):
    sample_category, sample_id = sample_name.split("_")
    room_shape, room_h = _get_room_params(
        osp.join(data_path, f"{sample_category}_specs.txt"), sample_name
    )

    camera_covering_samples, cam_index = _get_cam_covering_samples(
        osp.join(data_path, f"{sample_name}_cover.txt"), room_shape, cam_params
    )
    
    sample_coords: Tensor = _sample_index2coords(
        room_shape, room_h, camera_covering_samples
    )
    benchmark_visible_points = torch.zeros(
        (room_shape[0], room_shape[1], room_shape[2]), dtype=torch.float, device=DEVICE
    )
    benchmark_visible_points[
        sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]
    ] = 1

    if sample_category == "AC":
        spec_path = osp.join(data_path, "AC_specs.txt")
        if sample_id in [f"{i + 1:02d}" for i in range(9)]:
            cfg_path = "configs/ocp/_base_/std_surveil_AC_01to09.py"
        else:
            cfg_path = "configs/ocp/_base_/std_surveil_AC_10to32.py"
    else:
        raise NotImplementedError()
    room = _build_room(room_shape, room_h, cfg_path, cam_params)

    visible_points_coords = concat_points_with_color(room.visible_points, [0, 255, 0])
    benchmark_visible_points_coords = concat_points_with_color(
        benchmark_visible_points, [255, 0, 0]
    )
    intersection = concat_points_with_color(
        ((room.visible_points > 0) & (benchmark_visible_points > 0)),
        [255, 140, 0],
    )

    save_visualized_points(
        torch.cat(
            (
                visible_points_coords,
                benchmark_visible_points_coords,
                intersection,
            ) if intersection is not None else (
                visible_points_coords,
                benchmark_visible_points_coords,
            ),
            dim=0,
        ),
        save_path=f"output/data_check/{sample_name}/",
        default_filename=f"cam_{cam_index}",
    )
                

if __name__ == "__main__":
    args = parse_args()

    main(args.sample_name, args.data_path, args.cam)