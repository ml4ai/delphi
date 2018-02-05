import pandas
from typing import Dict, List, Optional
from indra.statements import Influence

Delta = Dict[Optional[str], Optional[int]]
GroupBy = pandas.core.groupby.DataFrameGroupBy
Data = List[Influence]
