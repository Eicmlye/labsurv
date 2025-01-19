import os
import os.path as osp
from typing import List

from labsurv.utils.string import CLEAR_FORMAT


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
