from typing import List

from labsurv.utils import CLEAR_FORMAT


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
