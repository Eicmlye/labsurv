import os.path as osp

from labsurv.physics import BaseRoom


def main():
    load_from = "output/room/BaseRoom.pkl"
    if osp.exists(load_from):
        room = BaseRoom(load_from=load_from)
        room.save("output/room/BaseRoom_copy.pkl")
        room.visualize("output/room/BaseRoom_copy.ply")
    else:
        room = BaseRoom(shape=[20, 30, 20])
        room.add_block([3, 2, 4])
        room.add_block([2, 7, 3], color="red", near_origin_vertex=[15, 20, 5])
        room.save("output/room")
        room.visualize("output/room")


if __name__ == "__main__":
    main()
