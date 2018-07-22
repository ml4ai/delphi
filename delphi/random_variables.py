from typing import Dict, List, Optional
from datetime import datetime

Delta = Dict[Optional[str], Optional[int]]


class RV(object):
    def __init__(self, name):
        self.name = name
        self.dataset = None

    def sample(self):
        return random.choice(dataset)


class LatentVar(RV):
    def __init__(self, name):
        super().__init__(name)
        self.partial_t = None

class Indicator(RV):
    def __init__(
            self,
            name,
            source: str = None,
            unit: str = None,
            mean: float = None,
            stdev: float = None,
            time: datetime = None,
            ):
        super().__init__(name)
        self.source = source
        self.unit = unit
        self.mean = mean
        self.stdev = stdev
        self.time = time
