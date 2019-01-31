import random
from typing import Dict, List, Optional, Union, Set
from datetime import datetime
from .db import engine

Delta = Dict[Optional[str], Optional[int]]

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

MONTH_DICT = {
    month_name: month_number
    for month_name, month_number in zip(MONTHS, range(1, 13))
}


class RV(object):
    def __init__(self, name):
        self.name = name
        self.value = None
        self.dataset = None

    def sample(self):
        return random.choice(self.dataset)


class LatentVar(RV):
    def __init__(self, name):
        super().__init__(name)
        self.partial_t = None


class Indicator(RV):
    """
    The Indicator class represents an abstraction of a concrete, tangible
    quantity that is in some way representative of a higher level concept (i.e.
    a node in an :class:`delphi.AnalysisGraph.AnalysisGraph` object.)

    Args:
        source: The source database (FAO, WDI, etc.)
        unit: The units of the indicator.
        mean:
            The mean value of the indicator (for performing conditional
            forecasting queries on the model.)
        value: The current value of the indicator (used while performing inference)
        stdev: The standard deviation of the indicator.
        time: The time corresponding to the parameterization of the indicator.
        aggaxes:
            A list of axes across which the indicator values have been
            aggregated. Examples: 'month', 'year', 'state', etc.
        aggregation_method:
            The method of aggregation across the aggregation axes. Currently
            defaults to 'mean'.
    """

    def __init__(
        self,
        name,
        source: Optional[str] = None,
        unit: Optional[str] = None,
        mean: Optional[float] = None,
        value: Optional[float] = None,
        stdev: Optional[float] = None,
        time: Optional[datetime] = None,
        aggaxes: List[str] = [],
        aggregation_method: str = "mean",
    ):
        super().__init__(name)
        self.source = source
        self.unit = unit
        self.mean = mean
        self.value = value
        self.stdev = stdev
        self.time = time
        self.aggaxes = aggaxes
        self.aggregation_method = aggregation_method

    def get_historical_distribution(
        self,
        country: Optional[str] = "South Sudan",
        state: Optional[str] = None,
        county: Optional[str] = None,
        month: Optional[int] = None,
        source: Optional[str] = None,
    ):
        varName = self.name.replace("_", " ")
        query_components = [
            f"select * from indicator where `Variable` like '{varName}'",
            "and `Value` is not null",
        ]
        if country is not None:
            query_components.append(f"and `Country` like '{country}'")
        if state is not None:
            query_components.append(f"and `State` like '{state}'")
        if county is not None:
            query_components.append(f"and `County` like '{county}'")
        if month is not None:
            month = MONTH_DICT.get(month, month)
            query_components.append(f"and `Month` like '{float(month)}'")
        if source is not None:
            query_components.append(f"and `Source` like '{source}'")
        query = " ".join(query_components)
        return [float(x["Value"]) for x in engine.execute(query)]
