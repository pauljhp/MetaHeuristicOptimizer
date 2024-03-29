import networkx as nx
import numpy as np
import pandas as pd
import os
from pathlib import Path
from ..optimizer.base import (
    SearchSpace, Solution, Arc, ArcConstraints, Variable)
from .. import utils
import networkx as nx
from typing import Tuple, List


cwd = Path(os.getcwd())

def get_search_space(dataloc: str) -> Tuple[SearchSpace, nx.Graph, List[Variable], pd.DataFrame]:
    datadir = cwd.joinpath("MetaHeuristicOptimizer/problems/data")
    dpath = datadir.joinpath(dataloc)
    df = pd.read_csv(dpath.as_posix(),
        header=0, index_col=0)

    df = df.astype(np.float64)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    net = nx.convert_matrix.from_pandas_adjacency(df)
    net = utils.Graph(graph=net)

    variables = []
    for i in range(len(net.nodes())):
        exec(f"""variable{i + 1} = Variable(name="variable{i + 1}", domain=net.nodes())""")
        variables.append(eval(f"variable{i + 1}"))

    arcs, constraints = [], dict()
    for var1, var2 in utils.sliding_window_iter(variables, 2):
        arc = Arc((var1, var2))
        arcs.append(arc)
        constraints[arc.name] = lambda arc: net.is_neighbor(arc[0].name, arc[1].name)
    returnarc = Arc((variables[-1], variables[0]))
    arcs.append(returnarc)
    constraints[returnarc.name] = lambda arc: net.is_neighbor(arc[0].name, arc[1].name)
    arcconstraints = ArcConstraints(constraints)
    searchspace = SearchSpace(variables=variables, constraints=arcconstraints)
    return searchspace, net, variables, df    

dpath = cwd.joinpath("MetaHeuristicOptimizer/problems/data/tsp_us_cities.csv")
df = pd.read_csv(dpath.as_posix(),
    header=0,
    index_col=0)

df = df.astype(np.float64)
net = nx.convert_matrix.from_pandas_adjacency(df)
net = utils.Graph(graph=net)


variables = []
for i in range(len(net.nodes())):
    exec(f"""variable{i + 1} = Variable(name="variable{i + 1}", domain=net.nodes())""")
    variables.append(eval(f"variable{i + 1}"))


arcs, constraints = [], dict()
for var1, var2 in utils.sliding_window_iter(variables, 2):
    arc = Arc((var1, var2))
    arcs.append(arc)
    constraints[arc.name] = lambda arc: net.is_neighbor(arc[0].name, arc[1].name)
returnarc = Arc((variables[-1], variables[0]))
arcs.append(returnarc)
constraints[returnarc.name] = lambda arc: net.is_neighbor(arc[0].name, arc[1].name)
arcconstraints = ArcConstraints(constraints)

searchspace = SearchSpace(variables=variables, constraints=arcconstraints)

