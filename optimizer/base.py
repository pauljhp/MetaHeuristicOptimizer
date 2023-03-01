from abc import ABC, abstractmethod, abstractclassmethod
from typing import (Callable, Optional, Union, Sequence, 
    LiteralString, Literal, List, Dict, Type, Any, NewType, TypeVar,
    Tuple, Collection)
import numpy as np
import pandas as pd
import networkx as nx
import warnings
import utils



class Optimizer(ABC):
    def __init__(self):
        super(ABC, self).__init__()

    @abstractmethod
    def solve(self):
        raise NotImplementedError

# class Value

class AllSet(set):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
    def __contains__(self, other) -> bool:
        return True
    def __len__(self):
        return float("inf")


class Domain(set):
    def __init__(self, 
                 possiblevalues: Union[Sequence[Any], None],
                 valuetype: Optional[Type]=None,
                ):
        """
        :param possiblevalues: if set to None, a special domain object
            will be created with no value restrictions.
        """
        self.valuetype = Type
        if possiblevalues is not None:
            if valuetype is not None:
                assert len(possiblevalues) == sum(
                    [isinstance(v, valuetype) for v in possiblevalues]), \
                    f"possiblevalues should all be of type {valuetype}"
            self.possiblevalues = set(possiblevalues)
        else:
            self.possiblevalues = AllSet()
        

    def __contains__(self, other: object) -> bool:
        return self.possiblevalues.__contains__(other)
    def __eq__(self, other):
        return self.possiblevalues == other
    def __ne__(self, other):
        return self.possiblevalues != other
    def __len__(self):
        return self.possiblevalues.__len__()
    def __add__(self, other):
        return self.possiblevalues.union(other)
    def union(self, other):
        return self.possiblevalues.union(other)

Value = NewType("Value", Any)
    

class Variable:
    def __init__(self,
                 name: Union[LiteralString, int], 
                 value: Optional[Value]=None, 
                 domain: Optional[Domain]=None):
        """defines a node with name, value and domain
        :param name: name of the node. if not specified, can take any value
        :param value: value of this node
        """
        self.__name__ = name
        if domain is not None:
            if value in domain:
                self._domain = domain
                self._value = value
            else:
                raise ValueError(f"value should be in domain! Got value: {value}, which is not in domain: {domain}\n")
        else:
            self._domain = Domain(None)
            self._value = None
    
    def __getattr__(self, attr) -> Union[Any, None]:
        if attr in self.__dict__:
            return self.__getattribute__(attr)
        return None

    def __eq__(self, other) -> bool:
        if "value" in other.__dict__:
            return self.value == other.value
        else:
            return self.value == other
    
    def __ne__(self, other) -> bool:
        if "value" in other.__dict__:
            return self.value != other.value
        else:
            return self.value != other


    @property
    def name(self):
        return self.__name__
    @name.setter
    def name(self, newname: Union[LiteralString, int]):
        self.__name__ = newname

    @property 
    def domain(self):
        return self._domain
    @domain.setter
    def domain(self, newdomain: Domain):
        if self.value in newdomain:
            self._domain = newdomain
        else:
            warnings.warn(f"New domain will invalidate the existing value! Got value: {self.value}, which is not in the new domain: {newdomain}\n")
            self._domain = newdomain

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, newvalue: Value):
        if newvalue in self.domain:
            self._value = newvalue
        else:
            raise ValueError(f"value should be in domain! Got value: {newvalue}, which is not in domain: {self.domain}\n")

Arc = NewType("Arc", Tuple[Any])
Constraint = TypeVar("Constraint", bound=Callable)


class ArcConstraints:
    def __init__(self, 
                 constraint_arcs: Dict[Arc, Constraint]
                 ):
        """constraint_arcs is a dictionary of Arcs and Constraints.
        Constraints are callables which take the arcs and return True or false
        """
        self.constraint_arcs = constraint_arcs

    def satisfied(self) -> bool:
        for arc, const in self.constraint_arcs.items():
            if not const(arc):
                return False
        return True




class SearchSpace:
    def __init__(self,
        variables: Collection[Variable],
        constraints: Collection[ArcConstraints]
        ):
        """
        :param variables: collection of the variables
        :param constraints: collection of arc constraints
        """
        self.variables = variables
        self.constraints = constraints

    # def _check_legality(self, solution):
    #     return self._legality_check_fn(solution)
    
class Solution(SearchSpace):
    def __init__(self,
                 variables,
                 constraints
                 ):
        super().__init__(variables, constraints)

    def __iter__(self):
        return self.variables.__iter__()
    
    @property
    def is_legal(self) -> bool:
        return self.constraints.satisfied()

# class Solution(SearchSpace):
#     def __init__(self,
#         search_space: SearchSpace,
#         objective_fn: Callable,
#         solution: List):
#         super(SearchSpace, self).__init__(search_space, objective_fn)
#         self.search_space = search_space
#         self.data = search_space.data
#         self._legality_check_fn = search_space._legality_check_fn
#         self.data_structure = search_space.data_structure
#         self._objective_fn = objective_fn
#         self.solution = solution
    
#     @property
#     def get_of(self):
#         return self._objective_fn(self.data)

#     @property
#     def is_legal(self):
#         return self._check_legality(self.solution)

#     def mutate(self, idx: Optional[Sequence[int]]):
#         """mutate the existing solution"""
#         raise NotImplementedError

#     def opt2permute(self, 
#         start_idx: int, 
#         end_idx: int, 
#         inplace: bool=True):
#         """use opt-2 to update solution, inplace by default"""
#         first, third = self.solution[: start_idx], self.solution[end_idx:]
#         middle = self.search_space[start_idx : end_idx]
#         new_solution = first + middle + third
#         if self._check_legality(new_solution):
#             if inplace:
#                 self.solution = new_solution
#             else:
#                 return new_solution
#         else:
#             raise ValueError("new solution is not legal")


