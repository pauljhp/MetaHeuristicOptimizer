import networkx as nx
import numpy as np
import pandas as pd
import os
from pathlib import Path


cwd = Path(os.getcwd())
dpath = cwd.joinpath("problems/data/tsp_us_cities.csv")
df = pd.read_csv(dpath.as_posix(),
    header=0,
    index_col=0)

df = df.astype(np.float64)
net = nx.convert_matrix.from_pandas_adjacency(df)
