import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script to run other script after certain time."
    )

    parser.add_argument(  # --time, -t
        "--time",
        "-t",
        type=str,
        default=None,
        help=(
            "Time stamp to activate running. Should be in the format of "
            "\"yy-mm-dd-hh-mm-ss\"."
        ),
    )
    parser.add_argument(  # --command, -c
        "--command", "-c", type=str, default=None, help="The command to run."
    )
    parser.add_argument(
        "--sleep-step", "-s", type=int, default=300, help="Check time step in seconds."
    )

    args = parser.parse_args()

    return args


def get_time_stamp():
    cur_time = time.localtime()
    return (
        f"{cur_time.tm_year % 100}-{cur_time.tm_mon:02d}-{cur_time.tm_mday:02d}"
        f"-{cur_time.tm_hour:02d}-{cur_time.tm_min:02d}-{cur_time.tm_sec:02d}"
    )


def check_time_stamp_format(time_stamp: str):
    assert len(time_stamp) == 17
    assert len(time_stamp.split("-")) == 6


def check_time_stamp_after(cur_time_stamp: str, target_time_stamp: str):
    check_time_stamp_format(cur_time_stamp)
    check_time_stamp_format(target_time_stamp)

    cur_time = list(map(int, "-".join(cur_time_stamp.split()).split("-")))
    target_time = list(map(int, "-".join(target_time_stamp.split()).split("-")))

    for cur, target in zip(cur_time, target_time):
        if cur == target:
            continue

        if cur < target:
            return False
        else:
            return True

    return False  # all equal, not after


def main():
    args = parse_args()

    while not check_time_stamp_after(get_time_stamp(), args.time):
        print(f"Checked at {get_time_stamp()}, not activated.")
        time.sleep(args.sleep_step)

    print(f"Activated at {get_time_stamp()}")
    os.system(args.command)


if __name__ == "__main__":
    main()
