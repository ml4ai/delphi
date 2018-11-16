import numpy as np
import pandas as pd
from scipy.stats import norm
from delphi.utils.fp import flatMap, ltake, iterate
from delphi.AnalysisGraph import AnalysisGraph
from delphi.execution import get_latent_state_components, emission_function
from typing import List, Dict


def sample_transition_matrix_from_gradable_adjective_prior(
    G: AnalysisGraph, delta_t: float = 1.0
) -> pd.DataFrame:
    """ Return a pandas DataFrame object representing a transition matrix for
    the DBN. """
    elements = get_latent_state_components(G)
    A = pd.DataFrame(np.identity(2 * len(G)), index=elements, columns=elements)
    for n in G.nodes:
        A[f"∂({n})/∂t"][n] = delta_t
    for e in G.edges(data=True):
        A[f"∂({e[0]})/∂t"][e[1]] = (
            e[2]["ConditionalProbability"].resample(1)[0][0] * delta_t
        )
    return A


def get_sequence_of_latent_states(
    A: pd.DataFrame, s0: pd.Series, n_steps: int
) -> List[np.ndarray]:
    """ Return a list of pandas Series objects corresponding to latent state
    vectors. """
    return ltake(
        n_steps,
        iterate(lambda s: pd.Series(A.values @ s.values, index=s0.index), s0),
    )


def create_observed_state(G: AnalysisGraph) -> Dict:
    """ Create a dict corresponding to an observed state vector. """
    return {
        n[0]: {ind.name: ind.value for ind in n[1]["indicators"].values()}
        for n in G.nodes(data=True)
    }


def sample_observed_state(G: AnalysisGraph, s: pd.Series) -> Dict:
    """ Sample an observed state given a latent state vector. """
    for n in G.nodes(data=True):
        for indicator in n[1]["indicators"].values():
            indicator.value = np.random.normal(
                s[n[0]] * indicator.mean, indicator.stdev
            )

    return create_observed_state(G)


def evaluate_prior_pdf(
    A: pd.DataFrame, G: AnalysisGraph, delta_t: float = 1.0
) -> float:
    return np.prod(
        [
            e[2]["ConditionalProbability"].evaluate(
                A[f"∂({e[0]})/∂t"][e[1]] / delta_t
            )
            for e in G.edges(data=True)
        ]
    )


def evaluate_likelihood_pdf(
    G: AnalysisGraph,
    latent_state_vectors: List[np.ndarray],
    observed_states: List[Dict],
) -> float:
    s0 = latent_state_vectors[0]
    _list = []
    for latent_state, observed_state in zip(
        latent_state_vectors, observed_states
    ):
        for n in G.nodes(data=True):
            for indicator, value in observed_state[n[0]].items():
                ind = n[1]["indicators"][indicator]
                _list.append(
                    norm.pdf(value, latent_state[n] * ind.mean, ind.stdev)
                )

    return np.prod(_list)
