import argparse
import os
import os.path as osp
import pickle
from typing import List

import torch
from labsurv.utils.surveillance import apply_colormap_to_list, save_visualized_points
from matplotlib.colors import LinearSegmentedColormap
from numpy import ndarray as array
from torch import Tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render visit counts to heatmap position-wise."
    )

    parser.add_argument("--pkl", type=str, help="Path of the pkl file.")
    parser.add_argument("--save", type=str, default=None, help="Path to save figures.")

    return parser.parse_args()


def get_latest_pkl(dir_name: str):
    assert osp.isdir(dir_name)

    filenames: List[str] = os.listdir(dir_name)
    latest_episode: int = 0
    latest_pkl: str = None
    for filename in filenames:
        if filename.endswith(".pkl") and filename.startswith(
            "env_reset_count_episode_"
        ):
            cur_episode = int(filename.split(".")[0].split("_")[-1])

            if latest_episode < cur_episode:
                latest_episode = cur_episode
                latest_pkl = filename

    if latest_pkl is None:
        raise ValueError(f"No valid pkl file found in {dir_name}.")

    return osp.join(dir_name, latest_pkl)


def main():
    args = parse_args()

    pkl_filename = (
        get_latest_pkl(args.pkl) if not args.pkl.endswith(".pkl") else args.pkl
    )
    if args.save is None:
        args.save = osp.dirname(pkl_filename)

    with open(pkl_filename, "rb") as f:
        dump_dict: array = pickle.load(f)

    visit_count: array = dump_dict["visit_count"]  # [N]
    candidates: Tensor = torch.tensor(  # [N, 3]
        dump_dict["candidates"], dtype=torch.float, device=torch.device("cuda")
    )

    colors = [
        "#ffffff",  # black
        "#000000",  # white
    ]
    colormap = LinearSegmentedColormap.from_list("custom", colors, N=256)
    colored_dist: Tensor = apply_colormap_to_list(  # [N, 3]
        visit_count,
        colormap=colormap,
        device=torch.device("cuda"),
        divide_by_max=True,
    )

    points_with_color = torch.cat((candidates, colored_dist), dim=1)  # [N, 6]
    save_visualized_points(points_with_color, args.save, default_filename="visit_count")


if __name__ == "__main__":
    main()
