import argparse
import os.path as osp

import numpy as np
import torch
from configs.runtime import DEVICE as DEVICE_STR
from labsurv.physics import SurveillanceRoom


DEVICE = torch.device(DEVICE_STR)


def _get_room_params(spec_path: str, sample_name: str):
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


def _build_room(cfg_path: str, spec_path: str, sample_name: str) -> SurveillanceRoom:
    full_shape, room_h = _get_room_params(spec_path, sample_name)
    room = SurveillanceRoom(
        device=DEVICE,
        cfg_path=cfg_path,
        shape=full_shape,
    )
    print("Building occupancy blocks...")

    print("Building permission blocks...")
    room.add_block(
        [full_shape[0], full_shape[1], 1],
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
        [full_shape[0], full_shape[1], room_h],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )

    return room


def main(sample_name: str, data_path: str):
    sample_name_info = sample_name.split("_")
    assert len(sample_name_info) == 2
    sample_category, sample_id = sample_name_info
    assert sample_category in ["AC"]
    assert sample_id in [f"{i + 1:02d}" for i in range(32)]

    if sample_category == "AC":
        spec_path = osp.join(data_path, "AC_specs.txt")
        if sample_id in [f"{i + 1:02d}" for i in range(9)]:
            cfg_path = "configs/ocp/_base_/std_surveil_AC_01to09.py"
        else:
            cfg_path = "configs/ocp/_base_/std_surveil_AC_10to32.py"
    else:
        raise NotImplementedError()

    room = _build_room(cfg_path, spec_path, sample_name)

    save_dir = f"output/{sample_name}"
    room.save(save_dir)
    room.visualize(save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark room builder.")

    parser.add_argument(  # --sample-name, -n
        "--sample-name", "-n", type=str, help="Name of the sample."
    )
    parser.add_argument(  # --data-path, -d
        "--data-path", "-d", type=str, help="Directory path of data files."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args.sample_name, args.data_path)
