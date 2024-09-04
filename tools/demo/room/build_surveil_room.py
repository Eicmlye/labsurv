import argparse

import numpy as np
import torch
from configs.runtime import DEVICE
from labsurv.physics import SurveillanceRoom


def demo():
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
    else:
        full_shape = [50, 50, 50]
        room = SurveillanceRoom(
            device=DEVICE,
            cfg_path=cfg_path,
            shape=full_shape,
        )
        print("Building occupancy blocks...")
        room.add_block(
            [1, 10, 10],
            displacement=torch.tensor([20, 0, 0], device=DEVICE),
        )
        print("Building permission blocks...")
        room.add_block(
            [10, 10, 10],
            point_type="install_permitted",
            displacement=torch.tensor([5, 5, 5], device=DEVICE),
        )

        print("Building monitored blocks...")
        resol_requirement = dict(
            h_res_req_min=500,
            h_res_req_max=1000,
            v_res_req_min=500,
            v_res_req_max=1000,
        )
        room.add_block(
            full_shape,
            point_type="must_monitor",
            displacement=torch.tensor([0, 0, 0], device=DEVICE),
            **resol_requirement,
        )

        print("Adding cameras...")
        room.add_cam(
            torch.tensor([10, 10, 10], device=DEVICE),
            torch.tensor([0, 0], dtype=torch.float, device=DEVICE),
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
    elif size == "tiny":
        full_shape = [10, 10, 10]
        room = SurveillanceRoom(
            device=DEVICE,
            cfg_path=cfg_path,
            shape=full_shape,
        )
        print("Building occupancy blocks...")
        room.add_block(
            [3, 3, 3],
            displacement=np.array([3, 3, 3]),
        )

        print("Building permission blocks...")
        room.add_block(
            [1, 10, 6],
            point_type="install_permitted",
            displacement=np.array([0, 0, 4]),
        )
        room.add_block(
            [10, 1, 6],
            point_type="install_permitted",
            displacement=np.array([0, 0, 4]),
        )
        # room.add_block(
        #     [10, 10, 1],
        #     point_type="install_permitted",
        #     displacement=np.array([0, 0, 0]),
        # )
        room.add_block(
            [1, 10, 6],
            point_type="install_permitted",
            displacement=np.array([9, 0, 4]),
        )
        room.add_block(
            [10, 1, 6],
            point_type="install_permitted",
            displacement=np.array([0, 9, 4]),
        )
        # room.add_block(
        #     [10, 10, 1],
        #     point_type="install_permitted",
        #     displacement=np.array([0, 0, 9]),
        # )

        print("Building monitored blocks...")
        resol_requirement = dict(
            h_res_req_min=500,
            h_res_req_max=1000,
            v_res_req_min=500,
            v_res_req_max=1000,
        )
        room.add_block(
            [2, 6, 6],
            point_type="must_monitor",
            displacement=np.array([1, 2, 0]),
            **resol_requirement,
        )
        room.add_block(
            [2, 6, 2],
            point_type="must_monitor",
            displacement=np.array([7, 1, 3]),
            **resol_requirement,
        )

        # print("Adding cameras...")
        # room.add_cam(
        #     np.array([5, 10, 7]),
        #     np.array([torch.pi / 2, 0], dtype=torch.float),
        #     "std_cam",
        # )

        room.save("output/surv_room")
        room.visualize("output/surv_room")
        # room.visualize("output/surv_room", "camera")
    elif size == "standard":
        full_shape = [30, 30, 30]
        room = SurveillanceRoom(
            device=DEVICE,
            cfg_path=cfg_path,
            shape=full_shape,
        )
        print("Building occupancy blocks...")
        room.add_block(
            [10, 1, 4],
            displacement=torch.tensor([5, 6, 2], device=DEVICE),
        )

        print("Building permission blocks...")
        room.add_block(
            [10, 5, 5],
            point_type="install_permitted",
            displacement=torch.tensor([2, 7, 4], device=DEVICE),
        )

        print("Building monitored blocks...")
        resol_requirement = dict(
            h_res_req_min=500,
            h_res_req_max=1000,
            v_res_req_min=500,
            v_res_req_max=1000,
        )
        room.add_block(
            [5, 10, 3],
            point_type="must_monitor",
            displacement=torch.tensor([0, 0, 0], device=DEVICE),
            **resol_requirement,
        )
        room.add_block(
            [10, 5, 7],
            point_type="must_monitor",
            displacement=torch.tensor([10, 1, 0], device=DEVICE),
            **resol_requirement,
        )
        room.add_block(
            [10, 8, 10],
            point_type="must_monitor",
            displacement=torch.tensor([2, 14, 2], device=DEVICE),
            **resol_requirement,
        )

        # print("Adding cameras...")
        # room.add_cam(
        #     torch.tensor([5, 10, 7], device=DEVICE),
        #     torch.tensor([torch.pi / 2, 0], dtype=torch.float, device=DEVICE),
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
        choices=["tiny", "standard"],
        default="tiny",
        help="Size of the room.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        demo()
    else:
        main(args.size)
