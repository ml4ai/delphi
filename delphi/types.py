import pandas
from typing import Dict, List, Optional, NewType
from indra.statements import Influence
import networkx
from flask import Flask
from pandas import Series
import json

Delta = Dict[Optional[str], Optional[int]]
GroupBy = pandas.core.groupby.DataFrameGroupBy
DiGraph = networkx.classes.digraph.DiGraph
