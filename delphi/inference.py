from dataclasses import dataclass
import numpy as np
import pandas as pd
import random
from math import log
from scipy.stats import norm
from delphi.utils.fp import flatMap, ltake, iterate
from delphi.AnalysisGraph import AnalysisGraph
from delphi.execution import (
    get_latent_state_components,
    emission_function,
    construct_default_initial_state,
)
from typing import List, Dict
from itertools import permutations


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


class Sampler:
    """ MCMC sampler. """

    def __init__(self, G: AnalysisGraph, observed_states: List[Dict]):
        self.G = G
        self.A = sample_transition_matrix_from_gradable_adjective_prior(G)
        self.observed_states = observed_states
        self.score: float = None
        self.candidate_score: float = None
        self.max_score: float = None
        self.delta_t = 1.0

        self.index_permutations = list(permutations(range(len(self.A)), 2))
        s0 = construct_default_initial_state(self.G)
        self.latent_states = get_sequence_of_latent_states(
            self.A, s0, len(self.observed_states)
        )
        self.log_prior = self.calculate_log_prior()
        self.log_likelihood = self.calculate_log_likelihood()
        self.log_joint_probability = self.log_prior + self.log_likelihood

    def calculate_log_prior(self) -> float:
        _list = [
            log(
                e[2]["ConditionalProbability"].evaluate(
                    self.A[f"∂({e[0]})/∂t"][e[1]] / self.delta_t
                )
            )
            for e in self.G.edges(data=True)
        ]

        return sum(_list)

    def calculate_log_likelihood(self) -> float:
        s0 = self.latent_states[0]
        _list = []
        for latent_state, observed_state in zip(
            self.latent_states, self.observed_states
        ):
            for n in self.G.nodes(data=True):
                for indicator, value in observed_state[n[0]].items():
                    ind = n[1]["indicators"][indicator]
                    log_likelihood = np.log(
                        norm.pdf(
                            value, latent_state[n[0]] * ind.mean, ind.stdev
                        )
                    )
                    _list.append(log_likelihood)

        return sum(_list)

    def calculate_log_joint_probability(self):
        return self.calculate_log_prior() + self.calculate_log_likelihood()

    def get_sample(self):
        # Choose the element of A to perturb
        i, j = random.choice(self.index_permutations)

        original_value = self.A.values[i][j]
        original_log_joint_probability = self.calculate_log_joint_probability()
        self.A.values[i][j] = np.random.normal(self.A.values[i][j], 0.1)
        candidate_log_joint_probability = (
            self.calculate_log_joint_probability()
        )

        log_probability_ratio = (
            candidate_log_joint_probability - original_log_joint_probability
        )
        acceptance_probability = min(1, np.exp(log_probability_ratio))
        if acceptance_probability > np.random.rand():
            return self.A
        else:
            self.A.values[i][j] = original_value
            return self.A
