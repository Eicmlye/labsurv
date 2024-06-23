from labsurv.models.envs import BaseRoom


def main():
    room = BaseRoom([20, 30, 20])
    room.add_block([3, 2, 4])
    room.add_block([2, 7, 3], color="red", near_origin_vertex=[15, 20, 5])
    room.save("output/room")
    room.save("output/room.ply")


if __name__ == "__main__":
    main()
