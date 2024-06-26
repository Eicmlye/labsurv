# modified from gym.utils.seeding
from typing import Any, Optional, Tuple

import numpy as np


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, Any]:
    """
    ## Description:

        Generates a random number generator from the seed and returns the Generator and seed.

    ## Arguments:

        seed: The seed used to create the generator

    ## Returns:

        The generator and resulting seed

    ## Raises:

        ValueError: Seed must be a non-negative integer or omitted
    """
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise ValueError(f"Seed must be a non-negative integer or omitted, not {seed}")

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed


RNG = RandomNumberGenerator = np.random.Generator
