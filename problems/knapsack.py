import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from ..optimizer.base import (
    SearchSpace, Solution, Arc, ArcConstraints, Variable, Domain)
from .. import utils
from typing import Optional, Union, Any, Literal


cwd = Path(os.getcwd())
ProblemSet = Literal["profit_maximization"]
ProblemName = Literal["problem1", "problem2"]

def get_search_space(
        problem_set: ProblemSet,
        problem_name: ProblemName) -> SearchSpace:
    datadir = cwd.joinpath("MetaHeuristicOptimizer/problems/data")
    dpath = datadir.joinpath(f"{problem_set}.json")
    with dpath.open("r") as f:
        data = json.load(f)
    problem_data = data.get(problem_name)
    variables = []
    domain = Domain(possiblevalues=(0, 1), valuetype=int)
    for i, project in enumerate(problem_data.get("projects")):
        exec(f"""project{i + 1} = Variable(name="project{i + 1}", value=0, domain=domain, attributes=project)""") # set every project to 0 (not included) by default
        variables.append(eval(f"project{i + 1}"))
    arc = Arc(tuple(variables))
    constraint = lambda arc: np.dot(np.array([var.attributes.get("cost") for var in arc]),
                                    np.array([var.value for var in arc]))  <= problem_data.get("capacity")
    arcconstraints = ArcConstraints({arc: constraint})
    searchspace = SearchSpace(variables=variables, constraints=arcconstraints)
    return searchspace