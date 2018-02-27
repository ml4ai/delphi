import pandas
from typing import Dict, List, Optional, NewType
from indra.statements import Influence
import networkx
from flask import Flask
from pandas import Series

class State(object):
    """ Class to hold the global state of the application """
    CAG                  : Optional[Dict]            = None
    factors              : Optional[List[str]]       = None
    s0                   : Optional[Series]       = None
    s_index              : Optional[List[str]]       = None
    initialValues        : Optional[List[float]]     = None
    elementsJSON         : Optional[Dict]            = None
    n_steps              : int                       = 10
    n_samples            : int                       = 10000
    statements           : Optional[List[Influence]] = None
    elementsJSONforJinja : Optional[Dict]            = None
    histos_built         : bool                      = False
    Δt                   : float                     = 1
    inputText            : str                       = ''

Delta = Dict[Optional[str], Optional[int]]
GroupBy = pandas.core.groupby.DataFrameGroupBy
Data = List[Influence]
DiGraph = networkx.classes.digraph.DiGraph
