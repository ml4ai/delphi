import pandas
from datetime import datetime
from typing import Dict, List, Optional, NewType
from indra.statements import Influence
import networkx
from pandas import Series
import json
from dataclasses import dataclass
import networkx as nx
import pandas as pd
from scipy.stats import gaussian_kde
from .utils import exists, flatMap
import numpy as np

Delta = Dict[Optional[str], Optional[int]]
GroupBy = pandas.core.groupby.DataFrameGroupBy
DiGraph = networkx.classes.digraph.DiGraph



class AnalysisGraph(DiGraph):
    pass


@dataclass
class Indicator:
    name: str
    source: str
    value: float = None
    stdev: float = None
    time: datetime = None

@dataclass(frozen=True)
class Node:
    name: str
    indicators: List[Indicator] = None
