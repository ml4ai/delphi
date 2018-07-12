from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

Delta = Dict[Optional[str], Optional[int]]

@dataclass
class Indicator:
    name: str
    source: str
    unit: str = None
    value: float = None
    stdev: float = None
    time: datetime = None
    relative_polarity = 1

@dataclass(frozen=True)
class Node:
    name: str
    indicators: List[Indicator] = None
