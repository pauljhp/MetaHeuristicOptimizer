import numpy as np
import math
import networkx as nx
from .base import Optimizer
from typing import Union, Optional, Tuple, List, Sequence, Callable
from argparse import ArgumentParser
from ..utils import get_rng
import heapq
from ..logger import Logger
# import math


DEFAULT_SEED = 123
RNG = np.random.default_rng(seed=DEFAULT_SEED)


class EquilibriumOptimizer(Optimizer):
    def __init__(self,
        population_size: int,
        max_iter: int,
        fitness_fn: Callable,
        search_space: Callable,
        seed: int,
        dim: int,
        runno: Optional[int]=None,
        logpath: Optional[str]=None):
        """minimizes by default"""
        self.population_size = population_size
        self.max_iter = max_iter
        self.fitness_fn = fitness_fn
        self.search_space = search_space
        self.rng = get_rng(seed)
        self.dim = dim
        self._population, self._fitness = [], []
        self._equilibrium_pool = [(float("-inf"), None)] * 4 # heap with the population index
        heapq.heapify(self._equilibrium_pool)
        self.c_min = np.array(self.search_space.min())
        self.c_max = np.array(self.search_space.max())
        self.runno = runno
        self.logpath = logpath
    
    def update_equilibrium_pool(self,
                                newcost,
                                newval_idx,) -> None:
        """top 4 values maintained in a heapq
        equilibrium_pool: heapified list of (cost, index)
        """
        min_val, _ = min(self._equilibrium_pool)
        idxs = [x for _, x in self._equilibrium_pool]
        if min_val < newcost and not (newval_idx in idxs):
            heapq.heappush(self._equilibrium_pool, (newcost, newval_idx))
            heapq.heappop(self._equilibrium_pool)

    def initialize(self):
        """call this first"""
        for i in range(self.population_size):
            rand_mask = self.rng.uniform(low=0., high=1., size=self.dim)
            self._population.append(
                self.c_min + (self.c_max - self.c_min) * rand_mask)
            self._fitness.append(self.fitness_fn(self._population[i]))
            self.update_equilibrium_pool(-self._fitness[i], i)
        self.population_ = self._population

    def optimize(self,
                alpha1: float=0.1,
                alpha2: float=0.1,
                gp: float=0.5,
                verbose: bool=False,
                log: bool=True
                ):
        if log:
            self.logger = Logger(params=dict(
                alpha1=alpha1,
                alpha2=alpha2,
                gp=gp,
                population_size=self.population_size,
                max_iter=self.max_iter),
                runno=self.runno,
                logpath=self.logpath if self.logpath else None)
        iterno = 1
        while iterno <= self.max_iter:
            if verbose:
                print(f"starting epoch {iterno}")
            
            t = (1 - iterno / self.max_iter) ** (alpha2 * iterno / self.max_iter)
            
            # update equilibrium pool
            for i in range(self.population_size):
                C = self.population_[i]
                C_fit = self.fitness_fn(C)
                self.update_equilibrium_pool(-C_fit, i)

            # update population
            for i in range(self.population_size):
                
                C = self.population_[i]
                rand_idx = self.rng.integers(low=0, high=4, size=1)[0]
                if rand_idx < 4:
                    C_eq_fitness, C_eq_idx  = self._equilibrium_pool[rand_idx]
                    C_eq = self.population_[C_eq_idx]
                else: # assign C_ave
                    C_ave = np.mean(self.population_[i] for i in [x for _, x in self._equilibrium_pool])
                    C_eq = C_ave
                    C_eq_fitness = np.mean([x for x, _ in self._equilibrium_pool])
                _lambda = self.rng.uniform(low=0., high=1., size=self.dim)
                rnd = self.rng.uniform(low=0., high=1., size=1)[0]
                F = alpha1 * np.sign(rnd - .5) * np.exp(- _lambda * t)
                r1 = self.rng.uniform(low=0., high=1., size=1)[0]
                r2 = self.rng.uniform(low=0., high=1., size=1)[0]
                GCP = 0.5 * r1 if r2 > gp else 0.
                G0 = GCP * (C_eq - _lambda * C)
                G = G0 * F
                self.population_[i] = C_eq + (C - C_eq) * F + G / _lambda * (1 - F)
            
            if verbose:
                print(f"finished epoch {iterno}.\n Best finess so far: {C_eq_fitness}")
            if log:
                self.logger.log("performance", {f"epoch {iterno}": C_eq_fitness})
            iterno += 1
        return self