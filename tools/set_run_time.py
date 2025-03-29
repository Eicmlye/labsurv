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
    parser.add_argument(  # --sleep-step, -s
        "--sleep-step", "-s", type=int, default=300, help="Check time step in seconds."
    )
    parser.add_argument(  # --log, -l
        "--log", "-l", type=str, default="output/shell/", help="Log directory."
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
    log_time = get_time_stamp()
    log_filename = (
        "_".join(["".join(log_time.split("-")[:3]), "".join(log_time.split("-")[3:])])
        + ".log"
    )
    log_path = os.path.join(args.log, log_filename)
    print(f"Log will be saved at {log_path}.")

    start_prompt = f"{"Start time":<12}{log_time}\n{"Run after":<12}{args.time}"
    print(start_prompt)
    if args.log is not None:
        os.makedirs(args.log, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(start_prompt + "\n")

    while not check_time_stamp_after(get_time_stamp(), args.time):
        check_prompt = f"{"Checked at":<12}{get_time_stamp()}, not activated."
        print("\r\033[K" + check_prompt, end="")

        if args.log is not None:
            with open(log_path, "a") as f:
                f.write(check_prompt + "\n")

        time.sleep(args.sleep_step)

    run_prompt = f"{"Activated":<12}{get_time_stamp()}"
    print("\r\033[K" + run_prompt)
    if args.log is not None:
        with open(log_path, "a") as f:
            f.write(run_prompt + "\n")
            f.write(f"Will run command \"{args.command}\"")

    os.system(args.command)


if __name__ == "__main__":
    main()
