import numpy as np
from scipy.stats import rankdata
from typing import Sequence, Optional, Iterable
import itertools
import networkx as nx
import re
import copy
import math
from typing import Tuple


EARTH_R = 6371.  # Earth radius in km


def sequence_to_rank(input: Sequence, 
                    expectedlen: Optional[int]=None):
    """convert a sequence into ranks
    e.g. [0.1, 0.9, 9, -0.1] would return [2, 3, 4, 1]
    """
    if expectedlen is not None:
        assert expectedlen == input, f"expected length must be the same as input! expected {expectedlen:.0f}, got {len(input)}"
    ranks = rankdata(input).astype(np.int64)
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
    

class Graph(nx.Graph):
    def __init__(self, 
                 graph: Optional[nx.Graph]=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if graph is not None:
            for attr in dir(graph):
                if not re.match("__([\w\d]+)__", attr): # if not dunder method
                    val_to_copy = copy.deepcopy(graph.__getattribute__(attr))
                    self.__setattr__(attr, val_to_copy)
    
    def is_neighbor(self, A, B) -> bool:
        return (B in self.neighbors(A))



def distance_on_earth(loc1: Tuple[float], loc2: Tuple[float]):
    """Calculate the distance between two locations in km using the Haversine formula."""

    lat1, lon1 = loc1
    lat2, lon2 = loc2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 \
        + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_R * c