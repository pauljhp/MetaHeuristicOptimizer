import numpy as np
import math
import networkx as nx
from .base import Optimizer
from typing import Union, Optional, Tuple, List, Sequence, Callable
from argparse import ArgumentParser
from ..utils import get_rng


DEFAULT_SEED = 123
RNG = np.random.default_rng(seed=DEFAULT_SEED)


class EquilibriumOptimizer(Optimizer):
    def __init__(self,
        population_size: int,
        max_iter: int,
        fitness_fn: Callable,
        search_space: Callable,
        seed: int,
        dim: int):
        self.population_size = population_size
        self.max_iter = max_iter
        self.fitness_fn = fitness_fn
        self.search_space = search_space
        self.rng = get_rng(seed)
        self.dim = dim
        self._population = []
        self.c_min = self.search_space.min()

    def initialize(self):
        for i in range(self.population_size):
            rand_mask = self.rng.uniform(low=0., high=1., size=self.dim)
            



    # FIXEME - to be completed