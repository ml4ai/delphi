import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from delphi.utils.fp import flatMap, ltake, iterate
from delphi.paths import adjectiveData
from delphi.AnalysisGraph import AnalysisGraph
from delphi.assembly import constructConditionalPDF, get_respdevs
from delphi.execution import get_latent_state_components
from typing import List


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


def sample_from_prior(G: AnalysisGraph, delta_t: float = 1.0) -> pd.DataFrame:
    elements = get_latent_state_components(G)
    A = pd.DataFrame(np.identity(2 * len(G)), index=elements, columns=elements)
    for e in G.edges(data=True):
        A[e[1]][f"∂({e[0]})/∂t"] = (
            e[2]["ConditionalProbability"].resample(1)[0][0] * delta_t
        )

    return A


def sample_sequence_of_latent_states(
    A: np.ndarray, s0: np.ndarray, n_steps: int
) -> List[np.ndarray]:
    return ltake(n_steps, iterate(lambda s: A @ s, s0))


def calculate_prior_probability(A, G):
    return np.product(
        [
            e[2]["ConditionalProbability"].evaluate(A[e[1]][f"∂({e[0]})/∂t"])
            for e in G.edges(data=True)
        ]
    )


# def calculate_likelihood(latent_states, indicator_values):
