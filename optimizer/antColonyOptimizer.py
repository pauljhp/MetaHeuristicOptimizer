import numpy as np
import math
import networkx as nx
from .base import Optimizer
from typing import Union, Optional, Tuple, List, Sequence, Callable, Literal, Set
from argparse import ArgumentParser
from ..utils import get_rng
from .. import utils
import heapq
from ..logger import Logger
from .base import SearchSpace
import pandas as pd


DEFAULT_SEED = 123
RNG = np.random.default_rng(seed=DEFAULT_SEED)

class AntColonyOptimizer:
    def __init__(self, search_space: Union[SearchSpace, nx.Graph], 
                 num_ants: int, 
                 alpha: float, beta: float, rho: float, q: float, max_iter: int):
        self.graph = search_space
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iter = max_iter
        self._current_iter = 0

    def solve(self, step_by_step: bool=False, step_size: int=1) -> Tuple[List[int], float]:
        best_path = None
        best_path_length = float("inf")

        self.pheromone_matrix_ = self.initialize_pheromone_matrix()

        if step_by_step:
            for _ in range(step_size):
                self._current_iter += 1
                print(f"running iteration {self._current_iter}")
                ant_paths = self.construct_ant_paths(self.pheromone_matrix_)
                self.pheromone_matrix_ = self.update_pheromone_matrix(self.pheromone_matrix_, ant_paths)

                shortest_path, shortest_path_length = self.find_shortest_path(ant_paths)
                if shortest_path_length < best_path_length:
                    best_path = shortest_path
                    best_path_length = shortest_path_length
        else:
            for i in range(self.max_iter):
                self._current_iter += 1
                print(f"running iterations {i}")
                ant_paths = self.construct_ant_paths(self.pheromone_matrix_)
                self.pheromone_matrix_ = self.update_pheromone_matrix(self.pheromone_matrix_, ant_paths)

                shortest_path, shortest_path_length = self.find_shortest_path(ant_paths)
                if shortest_path_length < best_path_length:
                    best_path = shortest_path
                    best_path_length = shortest_path_length

        return best_path, best_path_length

    def initialize_pheromone_matrix(self) -> pd.DataFrame:
        data = np.ones((len(self.graph.nodes), len(self.graph.nodes)))
        df = pd.DataFrame(data, index=list(self.graph.nodes), columns=list(self.graph.nodes))
        return df

    def construct_ant_paths(self, pheromone_matrix: np.ndarray) -> List[Tuple[List[int], float]]:
        ant_paths = []
        for _ in range(self.num_ants):
            start_node = RNG.choice(list(self.graph.nodes))
            path = self.construct_path(start_node, pheromone_matrix)
            ant_paths.append((path, self.path_length(path)))
        return ant_paths

    def construct_path(self, start_node: int, pheromone_matrix: np.ndarray) -> List[int]:
        path = [start_node]
        unvisited_nodes = set(self.graph.nodes) - {start_node}

        while unvisited_nodes:
            current_node = path[-1]
            next_node = self.choose_next_node(current_node, pheromone_matrix, unvisited_nodes) # choose randomly
            path.append(next_node)
            unvisited_nodes.remove(next_node)

        path.append(start_node)
        return path

    def choose_next_node(self, current_node: int, pheromone_matrix: np.ndarray, unvisited_nodes: Set[int]) -> int:
        probabilities = [
            (pheromone_matrix[current_node][node] ** self.alpha) *
            (self.graph[current_node][node]["weight"] ** -self.beta)
            for node in unvisited_nodes
        ]

        total_probability = sum(probabilities)
        normalized_probabilities = [prob / total_probability for prob in probabilities]

        return RNG.choice(list(unvisited_nodes), p=normalized_probabilities)

    def update_pheromone_matrix(self, pheromone_matrix: np.ndarray, ant_paths: List[Tuple[List[int], float]]) -> np.ndarray:
        delta_pheromone_matrix = pd.DataFrame(np.zeros_like(pheromone_matrix),
                                              index=pheromone_matrix.index,
                                              columns=pheromone_matrix.columns)

        for path, path_length in ant_paths:
            for i in range(len(path) - 1):
                delta_pheromone_matrix[path[i]][path[i+1]] += self.q / path_length

        return (1 - self.rho) * pheromone_matrix + delta_pheromone_matrix

    def find_shortest_path(self, ant_paths: List[Tuple[List[int], float]]) -> Tuple[List[int], float]:
        return min(ant_paths, key=lambda x: x[1])

    def path_length(self, path: List[int]) -> float:
        return sum(self.graph[path[i]][path[i+1]]["weight"] for i in range(len(path) - 1))