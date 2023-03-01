import numpy as np
from scipy.stats import rankdata
from typing import Sequence, Optional, Iterable
import itertools


def sequence_to_rank(input: Sequence, 
                    expectedlen: Optional[int]=None):
    """convert a sequence into ranks
    e.g. [0.1, 0.9, 9, -0.1] would return [2, 3, 4, 1]
    """
    if expectedlen is not None:
        assert expectedlen == vector, f"expected length must be the same as input! expected {expectedlen:.0f}, got {len(input)}"
    ranks = rankdata(input)
    return ranks

def get_rng(seed: int):
    return np.random.default_rng(seed=seed)
    
def iter_by_chunk(iterable: Iterable, chunk_size: int):
    """iterate by chunk size"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def sliding_window_iter(iterable: Iterable, window_size: int):
    if window_size <= len(iterable):
        i = 0
        while i <= len(iterable) - window_size:
            yield iterable[i: i + window_size]
            i += 1
    else:
        raise ValueError("window_size cannot be longer than iterable")