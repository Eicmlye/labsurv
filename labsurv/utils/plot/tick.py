from typing import List


def generate_absolute_ticks(
    start: int,
    end: int,
    step: int = 1,
    include_start: bool = True,
    include_end: bool = True,
) -> List[int]:
    """
    ## Description:

        some examples:
            >>> generate_absolute_ticks(1, 30, step=5)
            [1, 5, 10, 15, 20, 25, 30]

            >>> generate_absolute_ticks(1, 27, step=5)
            [1, 5, 10, 15, 20, 25, 27]

            >>> generate_absolute_ticks(2, 27, step=5)
            [2, 5, 10, 15, 20, 25, 27]

            >>> generate_absolute_ticks(11, 27, step=5)
            [11, 15, 20, 25, 27]

            >>> generate_absolute_ticks(0, 27, step=5)
            [0, 5, 10, 15, 20, 25, 27]
    """
    length = end - start + 1

    result = [start] if include_start else []
    result = result + [
        (start - start % step) + (i + 1) * step for i in range(length // step)
    ]
    if include_end and length % step != 0:
        result = result + [end]

    return result
