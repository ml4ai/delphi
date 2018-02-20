from indra.statements import Influence
from pandas import Series
from typing import Optional, Dict, List

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
