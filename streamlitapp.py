import os, sys
import streamlit as st
from pathlib import Path

cwd = Path(os.getcwd())

if cwd.parent.as_posix() not in sys.path:
    sys.path.append(cwd.parent.as_posix())

os.chdir("..")
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


from MetaHeuristicOptimizer.optimizer.base import (
    Solution, Variable, SearchSpace, ArcConstraints, Arc
)
from MetaHeuristicOptimizer.optimizer.equilibriumOptimizer import EquilibriumOptimizer
from MetaHeuristicOptimizer import utils
from MetaHeuristicOptimizer.problems import tsp


st.title("Equilibrium Optimizer demo")
choice = st.selectbox("Please select a problem"
             ("TSP - US 31 cities"))

match choice:
    case ["TSP - US 31 cities"]:
        searchspace = tsp.searchspace
st.write(os.getcwd())
st.write(searchspace)