import os.path as osp

from configs.runtime import DEVICE
from labsurv.physics import SurveillanceRoom
import torch
from torch import Tensor


def main():
    load_from = None
    load_from = "output/surv_room/SurveillanceRoom.pkl"

    cfg_path = "configs/surveillance/_base_/envs/std_surveil.py"
    if load_from is not None:
        room = SurveillanceRoom(
            device=DEVICE,
            load_from=load_from,
        )
        room.save("output/surv_room_copy/SurveillanceRoom.pkl")
        room.visualize("output/surv_room_copy/SurveillanceRoom_occ.ply")
        room.visualize("output/surv_room_copy/SurveillanceRoom_cam.ply", "camera")
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
            near_origin_vertex=torch.tensor([20, 0, 0], device=DEVICE),
        )
        print("Building permission blocks...")
        room.add_block(
            [10, 10, 10],
            point_type="install_permitted",
            near_origin_vertex=torch.tensor([5, 5, 5], device=DEVICE),
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
            near_origin_vertex=torch.tensor([0, 0, 0], device=DEVICE),
            **resol_requirement,
        )

        print("Adding cameras...")
        room.add_cam(
            torch.tensor([10, 10, 10], device=DEVICE),
            torch.tensor([0, 0], dtype=torch.float16, device=DEVICE),
            "std_cam",
        )

        room.save("output/surv_room")
        room.visualize("output/surv_room")
        room.visualize("output/surv_room", "camera")


if __name__ == "__main__":
    main()
