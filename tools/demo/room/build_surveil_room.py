import os.path as osp

from labsurv.physics import SurveillanceRoom


def main():
    load_from = "output/surv_room/SurveillanceRoom.pkl"
    cfg_path = "configs/surveillance/_base_/envs/std_surveil.py"
    if osp.exists(load_from):
        room = SurveillanceRoom(
            cfg_path=cfg_path,
            load_from=load_from,
        )
        room.save("output/surv_room/SurveillanceRoom_copy.pkl")
        room.visualize("output/surv_room/SurveillanceRoom_occ_copy.ply")
        room.visualize("output/surv_room/SurveillanceRoom_cam_copy.ply", "camera")
    else:
        room = SurveillanceRoom(
            cfg_path=cfg_path,
            shape=[20, 30, 20],
        )
        print("Building occupancy blocks...")
        room.add_block([1, 7, 7], near_origin_vertex=[10, 0, 0])
        print("Building permission blocks...")
        room.add_block(
            [1, 7, 7], near_origin_vertex=[5, 2, 0], point_type="install_permitted"
        )
        room.add_block(
            [2, 7, 3], point_type="install_permitted", near_origin_vertex=[15, 20, 5]
        )

        print("Building monitored blocks...")
        resol_requirement = dict(
            h_res_req_min=500,
            h_res_req_max=1000,
            v_res_req_min=500,
            v_res_req_max=1000,
        )
        room.add_block(
            [10, 20, 10],
            point_type="must_monitor",
            near_origin_vertex=[5, 5, 0],
            **resol_requirement,
        )

        print("Adding cameras...")
        room.add_cam([5, 7, 0], [0, 0], "std_cam")

        room.save("output/surv_room")
        room.visualize("output/surv_room")
        room.visualize("output/surv_room", "camera")


if __name__ == "__main__":
    main()
