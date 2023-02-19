import numpy as np
import math
import networkx as nx
from abstract import Optimizer
from typing import Union, Optional, Tuple, List, Sequence, Callable




class EquilibriumOptimizer(Optimizer):
    def __init__(self,
        population_size: int,
        max_iter: int,
        fitness_fn: Callable,
        search_space: Callable):
        pass
    # FIXEME - to be completed