import argparse
import os.path as osp
from typing import List

import numpy as np
import torch
from configs.runtime import DEVICE as DEVICE_STR
from labsurv.physics import SurveillanceRoom
from torch import pi as PI

DEVICE = torch.device(DEVICE_STR)


def _build_test_room(cfg_path: str) -> SurveillanceRoom:
    full_shape = [15, 30, 15]
    room = SurveillanceRoom(
        device=DEVICE,
        cfg_path=cfg_path,
        shape=full_shape,
    )
    print("Building occupancy blocks...")
    room.add_block(
        [5, 8, 15],
        displacement=np.array([5, 5, 0]),
    )
    room.add_block(
        [5, 8, 15],
        displacement=np.array([5, 17, 0]),
    )

    print("Building permission blocks...")
    room.add_block(
        [5, 30, 5],
        point_type="install_permitted",
        displacement=np.array([0, 0, 10]),
    )
    room.add_block(
        [5, 30, 5],
        point_type="install_permitted",
        displacement=np.array([10, 0, 10]),
    )
    room.add_block(
        [15, 5, 5],
        point_type="install_permitted",
        displacement=np.array([0, 0, 10]),
    )
    room.add_block(
        [15, 4, 5],
        point_type="install_permitted",
        displacement=np.array([0, 13, 10]),
    )
    room.add_block(
        [15, 5, 5],
        point_type="install_permitted",
        displacement=np.array([0, 25, 10]),
    )

    print("Building monitored blocks...")
    resol_requirement = dict(
        h_res_req_min=500,
        h_res_req_max=1000,
        v_res_req_min=500,
        v_res_req_max=1000,
    )
    room.add_block(
        [5, 30, 10],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [5, 30, 10],
        point_type="must_monitor",
        displacement=np.array([10, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [15, 5, 10],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [15, 4, 10],
        point_type="must_monitor",
        displacement=np.array([0, 13, 0]),
        **resol_requirement,
    )
    room.add_block(
        [15, 5, 10],
        point_type="must_monitor",
        displacement=np.array([0, 25, 0]),
        **resol_requirement,
    )

    return room


def _build_tiny_room(cfg_path: str) -> SurveillanceRoom:
    full_shape = [50, 50, 30]
    room = SurveillanceRoom(
        device=DEVICE,
        cfg_path=cfg_path,
        shape=full_shape,
    )
    print("Building occupancy blocks...")
    room.add_block(
        [20, 20, 30],
        displacement=np.array([15, 15, 0]),
    )

    print("Building permission blocks...")
    room.add_block(
        [15, 50, 10],
        point_type="install_permitted",
        displacement=np.array([0, 0, 20]),
    )
    room.add_block(
        [50, 15, 10],
        point_type="install_permitted",
        displacement=np.array([0, 0, 20]),
    )
    room.add_block(
        [15, 50, 10],
        point_type="install_permitted",
        displacement=np.array([35, 0, 20]),
    )
    room.add_block(
        [50, 15, 10],
        point_type="install_permitted",
        displacement=np.array([0, 35, 20]),
    )

    print("Building monitored blocks...")
    resol_requirement = dict(
        h_res_req_min=500,
        h_res_req_max=1000,
        v_res_req_min=500,
        v_res_req_max=1000,
    )
    room.add_block(
        [15, 50, 20],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [50, 15, 20],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [15, 50, 20],
        point_type="must_monitor",
        displacement=np.array([35, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [50, 15, 20],
        point_type="must_monitor",
        displacement=np.array([0, 35, 0]),
        **resol_requirement,
    )

    return room


def _build_standard_room(cfg_path: str) -> SurveillanceRoom:
    full_shape = [100, 100, 30]
    room = SurveillanceRoom(
        device=DEVICE,
        cfg_path=cfg_path,
        shape=full_shape,
    )

    print("Building occupancy blocks...")
    room.add_block(
        [20, 20, 30],
        displacement=np.array([20, 20, 0]),
    )
    room.add_block(
        [20, 20, 30],
        displacement=np.array([60, 60, 0]),
    )
    room.add_block(
        [20, 20, 30],
        displacement=np.array([20, 60, 0]),
    )
    room.add_block(
        [20, 20, 30],
        displacement=np.array([60, 20, 0]),
    )

    print("Building permission blocks...")
    room.add_block(
        [20, 100, 10],
        point_type="install_permitted",
        displacement=np.array([0, 0, 20]),
    )
    room.add_block(
        [100, 20, 10],
        point_type="install_permitted",
        displacement=np.array([0, 0, 20]),
    )
    room.add_block(
        [20, 100, 10],
        point_type="install_permitted",
        displacement=np.array([40, 0, 20]),
    )
    room.add_block(
        [100, 20, 10],
        point_type="install_permitted",
        displacement=np.array([0, 40, 20]),
    )
    room.add_block(
        [20, 100, 10],
        point_type="install_permitted",
        displacement=np.array([80, 0, 20]),
    )
    room.add_block(
        [100, 20, 10],
        point_type="install_permitted",
        displacement=np.array([0, 80, 20]),
    )

    print("Building monitored blocks...")
    resol_requirement = dict(
        h_res_req_min=50,
        h_res_req_max=1000,
        v_res_req_min=50,
        v_res_req_max=1000,
    )
    room.add_block(
        [20, 100, 20],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [20, 100, 20],
        point_type="must_monitor",
        displacement=np.array([40, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [20, 100, 20],
        point_type="must_monitor",
        displacement=np.array([80, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [100, 20, 20],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [100, 20, 20],
        point_type="must_monitor",
        displacement=np.array([0, 40, 0]),
        **resol_requirement,
    )
    room.add_block(
        [100, 20, 20],
        point_type="must_monitor",
        displacement=np.array([0, 80, 0]),
        **resol_requirement,
    )

    return room


def _build_empty_room(cfg_path: str) -> SurveillanceRoom:
    full_shape = [15, 15, 7]
    room = SurveillanceRoom(
        device=DEVICE,
        cfg_path=cfg_path,
        shape=full_shape,
    )
    print("Building occupancy blocks...")

    print("Building permission blocks...")
    room.add_block(
        [15, 15, 2],
        point_type="install_permitted",
        displacement=np.array([0, 0, 5]),
    )

    print("Building monitored blocks...")
    resol_requirement = dict(
        h_res_req_min=500,
        h_res_req_max=1000,
        v_res_req_min=500,
        v_res_req_max=1000,
    )
    room.add_block(
        [15, 15, 5],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )

    return room


def _build_huge_room(cfg_path: str, id: List[int]) -> SurveillanceRoom:
    full_shape = [100, 100, 30]
    room = SurveillanceRoom(
        device=DEVICE,
        cfg_path=cfg_path,
        shape=full_shape,
    )

    print("Building occupancy blocks...")
    resol_requirement = dict(
        h_res_req_min=50,
        h_res_req_max=1000,
        v_res_req_min=50,
        v_res_req_max=1000,
    )
    start_shifts = [
        [10, 10, 0],
        [10, 40, 0],
        [10, 70, 0],
        [40, 10, 0],
        [40, 40, 0],
        [40, 70, 0],
        [70, 10, 0],
        [70, 40, 0],
        [70, 70, 0],
    ]
    for num in range(1, 10):
        if num in id:
            room.add_block(
                [20, 20, 30],
                displacement=np.array(start_shifts[num - 1]),
            )
        else:
            room.add_block(
                [20, 20, 10],
                point_type="install_permitted",
                displacement=np.array(start_shifts[num - 1][:-1] + [20]),
            )
            room.add_block(
                [20, 20, 20],
                point_type="must_monitor",
                displacement=np.array(start_shifts[num - 1]),
                **resol_requirement,
            )

    print("Building permission blocks...")
    room.add_block(
        [10, 100, 10],
        point_type="install_permitted",
        displacement=np.array([0, 0, 20]),
    )
    room.add_block(
        [10, 100, 10],
        point_type="install_permitted",
        displacement=np.array([30, 0, 20]),
    )
    room.add_block(
        [10, 100, 10],
        point_type="install_permitted",
        displacement=np.array([60, 0, 20]),
    )
    room.add_block(
        [10, 100, 10],
        point_type="install_permitted",
        displacement=np.array([90, 0, 20]),
    )
    room.add_block(
        [100, 10, 10],
        point_type="install_permitted",
        displacement=np.array([0, 0, 20]),
    )
    room.add_block(
        [100, 10, 10],
        point_type="install_permitted",
        displacement=np.array([0, 30, 20]),
    )
    room.add_block(
        [100, 10, 10],
        point_type="install_permitted",
        displacement=np.array([0, 60, 20]),
    )
    room.add_block(
        [100, 10, 10],
        point_type="install_permitted",
        displacement=np.array([0, 90, 20]),
    )

    print("Building monitored blocks...")
    room.add_block(
        [10, 100, 20],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [10, 100, 20],
        point_type="must_monitor",
        displacement=np.array([30, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [10, 100, 20],
        point_type="must_monitor",
        displacement=np.array([60, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [10, 100, 20],
        point_type="must_monitor",
        displacement=np.array([90, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [100, 10, 20],
        point_type="must_monitor",
        displacement=np.array([0, 0, 0]),
        **resol_requirement,
    )
    room.add_block(
        [100, 10, 20],
        point_type="must_monitor",
        displacement=np.array([0, 30, 0]),
        **resol_requirement,
    )
    room.add_block(
        [100, 10, 20],
        point_type="must_monitor",
        displacement=np.array([0, 60, 0]),
        **resol_requirement,
    )
    room.add_block(
        [100, 10, 20],
        point_type="must_monitor",
        displacement=np.array([0, 90, 0]),
        **resol_requirement,
    )

    return room


def main(size: str, is_demo: bool, id: List[int]):
    load_from = None
    # load_from = "output/demo/surv_room/SurveillanceRoom.pkl"

    cfg_path = f"configs/ocp/_base_/std_surveil_{size}.py"
    if load_from is not None:
        room = SurveillanceRoom(
            device=DEVICE,
            load_from=load_from,
        )
        save_dir = "output/" + ("demo/" if is_demo else "") + "surv_room_copy"
        room.save(osp.join(save_dir, "SurveillanceRoom.ply"))
        room.visualize(osp.join(save_dir, "SurveillanceRoom_occ.ply"))
        if is_demo:
            room.visualize(osp.join(save_dir, "SurveillanceRoom_cam.ply"), "camera")

        return

    if size == "test":
        room = _build_test_room(cfg_path)

        if is_demo:
            print("Adding cameras...")
            room.add_cam(
                [2, 2, 12],
                [PI / 2, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [12, 2, 12],
                [PI / 2, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [2, 27, 12],
                [-PI / 2, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [12, 27, 12],
                [-PI / 2, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [5, 15, 12],
                [0, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [9, 15, 12],
                [-PI, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [5, 2, 12],
                [0, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [9, 2, 12],
                [-PI, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [5, 27, 12],
                [0, -PI * 5 / 18],
                "std_cam",
            )
            room.add_cam(
                [9, 27, 12],
                [-PI, -PI * 5 / 18],
                "std_cam",
            )
    elif size == "tiny":
        room = _build_tiny_room(cfg_path)

        if is_demo:
            print("Adding cameras...")
            room.add_cam(
                [5, 12, 25],
                [PI / 2, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [0, 45, 25],
                [0, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [44, 37, 25],
                [-PI / 2, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [49, 4, 25],
                [-PI, -PI / 6],
                "std_cam",
            )
    elif size == "standard":
        room = _build_standard_room(cfg_path)

        if is_demo:
            print("Adding cameras...")
            room.add_cam(
                [40, 10, 25],
                [0, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [60, 10, 25],
                [-PI, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [40, 50, 25],
                [0, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [60, 50, 25],
                [-PI, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [40, 90, 25],
                [0, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [60, 90, 25],
                [-PI, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [50, 40, 25],
                [PI / 2, -PI / 6],
                "std_cam",
            )
            room.add_cam(
                [50, 60, 25],
                [-PI / 2, -PI / 6],
                "std_cam",
            )
    elif size == "empty":
        room = _build_empty_room(cfg_path)

        if is_demo:
            print("Adding cameras...")
            room.add_cam(
                [0, 0, 6],
                [PI / 4, -PI / 4],
                "std_cam",
            )
    elif size == "huge":
        room = _build_huge_room(cfg_path, id)

        if is_demo:
            print("Adding cameras...")
            room.add_cam(
                [50, 90, 25],
                [-PI / 2, -PI / 6],
                "std_cam",
            )

    save_dir = "output/" + ("demo/" if is_demo else "") + f"surv_room_{size}"
    room.save(save_dir)
    room.visualize(save_dir)
    if is_demo:
        room.visualize(save_dir, "camera", heatmap=True)

        print(f"cov = {room.coverage}")


def parse_args():
    parser = argparse.ArgumentParser(description="SurveillanceRoom builder.")

    parser.add_argument("--demo", action="store_true", help="Build demo room.")
    parser.add_argument("--id", type=str, default="", help="obs ids.")
    parser.add_argument(  # --size
        "--size",
        choices=["test", "tiny", "standard", "empty", "huge"],
        default="tiny",
        help="Size of the room.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    id = [int(s) for s in args.id]

    main(args.size, args.demo, id)
