import argparse

import numpy as np
import torch
from configs.runtime import DEVICE as DEVICE_STR
from labsurv.physics import SurveillanceRoom
from torch import pi as PI

DEVICE = torch.device(DEVICE_STR)


def demo(size: str):
    load_from = None
    # load_from = "output/demo/surv_room/SurveillanceRoom.pkl"

    cfg_path = "configs/surveillance/_base_/envs/std_surveil.py"
    if load_from is not None:
        room = SurveillanceRoom(
            device=DEVICE,
            load_from=load_from,
        )
        room.save("output/demo/surv_room_copy/SurveillanceRoom.pkl")
        room.visualize("output/demo/surv_room_copy/SurveillanceRoom_occ.ply")
        room.visualize("output/demo/surv_room_copy/SurveillanceRoom_cam.ply", "camera")
    elif size == "test":
        full_shape = [50, 50, 30]
        room = SurveillanceRoom(
            device=DEVICE,
            cfg_path=cfg_path,
            shape=full_shape,
        )
        print("Building occupancy blocks...")
        room.add_block(
            [2, 2, 2],
            displacement=np.array([15, 15, 14]),
        )

        print("Building permission blocks...")
        room.add_block(
            [1, 1, 1],
            point_type="install_permitted",
            displacement=np.array([10, 10, 25]),
        )

        print("Building monitored blocks...")
        resol_requirement = dict(
            h_res_req_min=500,
            h_res_req_max=1000,
            v_res_req_min=500,
            v_res_req_max=1000,
        )
        room.add_block(
            [1, 1, 1],
            point_type="must_monitor",
            displacement=np.array([10, 10, 15]),
            **resol_requirement,
        )

        print("Adding cameras...")
        room.add_cam(
            [10, 10, 25],
            [0, -PI / 2],
            "std_cam",
        )

        room.save("output/demo/surv_room")
        room.visualize("output/demo/surv_room")
        room.visualize("output/demo/surv_room", "camera")
    elif size == "tiny":
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

        print("Adding cameras...")
        room.add_cam(
            [10, 15, 25],
            [PI / 4, -PI / 6],
            "std_cam",
        )

        room.save("output/demo/surv_room")
        room.visualize("output/demo/surv_room")
        room.visualize("output/demo/surv_room", "camera")
    elif size == "standard":
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

        print("Adding cameras...")
        room.add_cam(
            [50, 40, 25],
            [PI / 4, -PI / 6],
            "std_cam",
        )

        room.save("output/demo/surv_room")
        room.visualize("output/demo/surv_room")
        room.visualize("output/demo/surv_room", "camera")


def main(size: str):
    load_from = None
    # load_from = "output/surv_room/SurveillanceRoom.pkl"

    cfg_path = "configs/surveillance/_base_/envs/std_surveil.py"
    if load_from is not None:
        room = SurveillanceRoom(
            device=DEVICE,
            load_from=load_from,
        )
        room.save("output/surv_room_copy/SurveillanceRoom.pkl")
        room.visualize("output/surv_room_copy/SurveillanceRoom_occ.ply")
    elif size == "test":
        full_shape = [50, 50, 30]
        room = SurveillanceRoom(
            device=DEVICE,
            cfg_path=cfg_path,
            shape=full_shape,
        )
        print("Building occupancy blocks...")
        room.add_block(
            [10, 10, 30],
            displacement=np.array([20, 20, 0]),
        )

        print("Building permission blocks...")
        room.add_block(
            [20, 50, 10],
            point_type="install_permitted",
            displacement=np.array([0, 0, 20]),
        )
        room.add_block(
            [50, 20, 10],
            point_type="install_permitted",
            displacement=np.array([0, 0, 20]),
        )
        room.add_block(
            [20, 50, 10],
            point_type="install_permitted",
            displacement=np.array([30, 0, 20]),
        )
        room.add_block(
            [50, 20, 10],
            point_type="install_permitted",
            displacement=np.array([0, 30, 20]),
        )

        print("Building monitored blocks...")
        resol_requirement = dict(
            h_res_req_min=500,
            h_res_req_max=1000,
            v_res_req_min=500,
            v_res_req_max=1000,
        )
        room.add_block(
            [20, 50, 20],
            point_type="must_monitor",
            displacement=np.array([0, 0, 0]),
            **resol_requirement,
        )
        room.add_block(
            [50, 20, 20],
            point_type="must_monitor",
            displacement=np.array([0, 0, 0]),
            **resol_requirement,
        )
        room.add_block(
            [20, 50, 20],
            point_type="must_monitor",
            displacement=np.array([30, 0, 0]),
            **resol_requirement,
        )
        room.add_block(
            [50, 20, 20],
            point_type="must_monitor",
            displacement=np.array([0, 30, 0]),
            **resol_requirement,
        )

        # print("Adding cameras...")
        # room.add_cam(
        #     [10, 15, 25],
        #     [PI / 4, -PI / 6],
        #     "std_cam",
        # )

        room.save("output/surv_room")
        room.visualize("output/surv_room")
        # room.visualize("output/surv_room", "camera")
    elif size == "tiny":
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

        # print("Adding cameras...")
        # room.add_cam(
        #     [10, 15, 25],
        #     [PI / 4, -PI / 6],
        #     "std_cam",
        # )

        room.save("output/surv_room")
        room.visualize("output/surv_room")
        # room.visualize("output/surv_room", "camera")
    elif size == "standard":
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

        # print("Adding cameras...")
        # room.add_cam(
        #     [50, 40, 25],
        #     [PI / 4, -PI / 6],
        #     "std_cam",
        # )

        room.save("output/surv_room")
        room.visualize("output/surv_room")
        # room.visualize("output/surv_room", "camera")


def parse_args():
    parser = argparse.ArgumentParser(description="SurveillanceRoom builder.")

    parser.add_argument("--demo", action="store_true", help="Build demo room.")
    parser.add_argument(  # --size
        "--size",
        choices=["test", "tiny", "standard"],
        default="tiny",
        help="Size of the room.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        demo(args.size)
    else:
        main(args.size)
