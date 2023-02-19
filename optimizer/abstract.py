from abc import ABC, abstractmethod, abstractclassmethod
from typing import Callable


class Optimizer(ABC):
    def __init__(self):
        super(ABC, self).__init__()

    @abstractmethod
    def solve(self):
        raise NotImplementedError


class SearchSpace:
    def __init__(self,
        legality_check_fn: Callable):
        """
        :param search_type: takes "constrained" and "unconstrained"
        """
        self._legality_check_fn = legality_check_fn

    def _check_legality(self, solution):
        return self._legality_check_fn(solution)


class Solution(SearchSpace):
    def __init__(self,
        data: Callable,
        legality_check_fn: Callable,
        objective_fn: Callable):
        super(SearchSpace, self).__init__(data, legality_check_fn)
        self._objective_fn = objective_fn
    
    @property
    def get_of(self):
        return self._objective_fn(self.data)

    @property
    def is_legal(self):
        return self._check_legality()