import pandas as pd
from .AnalysisGraph import AnalysisGraph
from datetime import datetime
from typing import Optional
from .assembly import get_indicator_value, get_data
from .paths import south_sudan_data


def parameterize(
    G: AnalysisGraph, time: datetime, data=south_sudan_data
) -> AnalysisGraph:
    """ Parameterize the analysis graph.

    Args:
        G
        time
        datafile
    """

    if isinstance(data, str):
        G.data = get_data(data)
    else:
        G.data=data


    nodes_with_indicators = [
        n for n in G.nodes(data=True) if n[1]["indicators"] is not None
    ]

    for n in nodes_with_indicators:
        for indicator in n[1]["indicators"]:
            indicator.mean, indicator.unit = get_indicator_value(
                indicator, time, G.data
            )
            indicator.time = time
            if not indicator.mean is None:
                indicator.stdev = 0.1 * abs(indicator.mean)

        n[1]["indicators"] = [
            ind for ind in n[1]["indicators"] if ind.mean is not None
        ]
    return G
