import os
import os.path as osp
from typing import List

from labsurv.utils.string import CLEAR_FORMAT, INDENT
import numpy as np
from numpy import ndarray as array


def color_str(color: str, content: List[str], seperator: str = " "):
    color_map = dict(
        red=["\033[31m", CLEAR_FORMAT],
        green=["\033[32m", CLEAR_FORMAT],
        yellow=["\033[33m", CLEAR_FORMAT],
        blue=["\033[34m", CLEAR_FORMAT],
    )

    color = color.lower()

    if color in color_map.keys():
        content = seperator.join(content)
        return color_map[color][0] + content + color_map[color][1]

    raise ValueError(f"Unknown color \"{color}\"")


def WARN(content: str, prefix: str = None):
    prefix = "WARNING" if prefix is None else prefix
    prefix = "[" + prefix + "]"

    return color_str("yellow", [prefix, content])


def FAIL(content: str, prefix: str = None):
    prefix = "FAILURE" if prefix is None else prefix
    prefix = "[" + prefix + "]"

    return color_str("red", [prefix, content])


def PASS(content: str, prefix: str = None):
    prefix = "SUCCESS" if prefix is None else prefix
    prefix = "[" + prefix + "]"

    return color_str("green", [prefix, content])


def INFO(content: str, prefix: str = None):
    prefix = "INFO" if prefix is None else prefix
    prefix = "[" + prefix + "]"

    return color_str("blue", [prefix, content])


def to_filename(path: str, expected_extension: str, default_filename: str) -> str:
    """
    ## Description:

        Automatically change `path` to an expected filename.

        - If `path` is an expected file, i.e., ended with `expected_extension`,
        all the parent directories will be made, and `path` is returned.
        - If `path is not ended with `expected_extension`, then it is considered as a
        directory. This directory will be made along with all the parent directories,
        and the default filename is returned with `path/` as prefix.

    ## Arguments:

        path (str): the input file path.

        expected_extension (str): the expected extension for the file.

        default_filename (str): the default filename WITHOUT EXTENSION when `path` is a
        directory.
    """
    if not expected_extension.startswith("."):
        expected_extension = "." + expected_extension

    if not path.endswith(expected_extension):
        os.makedirs(path, exist_ok=True)
        result = osp.join(path, default_filename + expected_extension)
    else:
        os.makedirs(osp.dirname(path), exist_ok=True)
        result = path

    return result


def readable_param(param: array) -> str:
    """
    ## Arguments:

        param (np.ndarray): [PARAM_DIM]

    ## Returns:

        readable (str)
    """
    cache: List[array] = [
        param[:3].astype(np.int64),
        param[3:5].astype(np.float32),
        param[5:].nonzero()[0].astype(np.int64),
    ]

    cache[1] = cache[1] / np.pi * 180

    readable_list: List = []
    for i in range(len(cache)):
        readable_list += cache[i].tolist()

    readable: str = "["
    for i in range(3):
        readable += f"{readable_list[i]:>3d}" + INDENT
    for i in range(3, 5):
        readable += f"{readable_list[i]:>7.2f}" + INDENT
    readable += str(readable_list[-1]) + "]"

    return readable


def readable_action(action: array):
    """
    ## Arguments:

        action (np.ndarray): [ACTION_DIM]

    ## Returns:

        readable (str)
    """
    action_index: int = action.nonzero()[0].astype(np.int64)[0]

    if action_index < 2:
        readable = " " * 2 + ("-" if action_index % 2 == 0 else "+") + "x" + " " * 32
    elif action_index < 4:
        readable = " " * 7 + ("-" if action_index % 2 == 0 else "+") + "y" + " " * 27
    elif action_index < 6:
        readable = " " * 12 + ("-" if action_index % 2 == 0 else "+") + "z" + " " * 22
    elif action_index < 8:
        readable = " " * 21 + ("-" if action_index % 2 == 0 else "+") + "p" + " " * 13
    elif action_index < 10:
        readable = " " * 30 + ("-" if action_index % 2 == 0 else "+") + "t" + " " * 4
    else:
        readable = " " * 32 + "->" + str(action_index - 10)

    return readable
