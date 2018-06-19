import pandas
from datetime import datetime
from typing import Dict, List, Optional, NewType
from indra.statements import Influence
import networkx
from pandas import Series
import json
from dataclasses import dataclass

Delta = Dict[Optional[str], Optional[int]]
GroupBy = pandas.core.groupby.DataFrameGroupBy
DiGraph = networkx.classes.digraph.DiGraph

class CausalAnalysisGraph(DiGraph):
    pass

@dataclass
class Indicator:
    name: str
    source: str
    value: float = None
    stdev: float = None

@dataclass(frozen=True)
class Node:
    name: str
    indicators: List[Indicator] = None
