import pandas
from typing import Dict, List, Optional, NewType
from indra.statements import Influence
import networkx


Delta = Dict[Optional[str], Optional[int]]
GroupBy = pandas.core.groupby.DataFrameGroupBy
Data = List[Influence]
DiGraph = networkx.classes.digraph.DiGraph
