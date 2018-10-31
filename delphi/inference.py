import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from delphi.utils.fp import flatMap
from delphi.paths import adjectiveData
from delphi.AnalysisGraph import AnalysisGraph
from delphi.assembly import constructConditionalPDF, get_respdevs


def infer_transition_model(
    G: AnalysisGraph, adjective_data: str = None, res: int = 100
):
    """ Add probability distribution functions constructed from gradable
    adjective data to the edges of the analysis graph data structure.

    Args:
        adjective_data
        res
    """

    G.res = res
    if adjective_data is None:
        adjective_data = adjectiveData

    gb = pd.read_csv(adjective_data, delim_whitespace=True).groupby(
        "adjective"
    )

    rs = gaussian_kde(
        flatMap(
            lambda g: gaussian_kde(get_respdevs(g[1]))
            .resample(res)[0]
            .tolist(),
            gb,
        )
    ).resample(res)[0]

    for e in G.edges(data=True):
        e[2]["ConditionalProbability"] = constructConditionalPDF(gb, rs, e)
        e[2]["betas"] = np.tan(
            e[2]["ConditionalProbability"].resample(G.res)[0]
        )
