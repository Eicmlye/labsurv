import os.path as osp

from labsurv.utils.surveillance import SurveillanceRoom


def main():
    load_from = "output/surv_room/SurveillanceRoom.pkl"
    cfg_path = "configs/surveillance/_base_/envs/std_surveil.py"
    if osp.exists(load_from):
        room = SurveillanceRoom(
            cfg_path=cfg_path,
            load_from=load_from,
        )
        room.save("output/surv_room/SurveillanceRoom_copy.pkl")
        room.visualize("output/surv_room/SurveillanceRoom_copy.ply")
    else:
        room = SurveillanceRoom(
            cfg_path=cfg_path,
            shape=[20, 30, 20],
        )
        room.add_block([3, 2, 4])
        room.add_block([3, 2, 4], point_type="install_permitted")
        room.add_block(
            [2, 7, 3], point_type="install_permitted", near_origin_vertex=[15, 20, 5]
        )

        resol_requirement = dict(
            horizontal=5,
            vertical=5,
        )
        room.add_block(
            [5, 2, 4],
            point_type="must_monitor",
            near_origin_vertex=[7, 10, 15],
            **resol_requirement,
        )

        room.add_cam([1, 1, 3], [0.6, 0.6], "demo_cam")

        room.save("output/surv_room")
        room.visualize("output/surv_room")


if __name__ == "__main__":
    main()
