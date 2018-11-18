from dataclasses import dataclass
import numpy as np
import pandas as pd
import random
from math import log
from scipy.stats import norm
from delphi.utils.fp import flatMap, ltake, iterate
from delphi.AnalysisGraph import AnalysisGraph
from typing import List, Dict
from itertools import permutations


def create_observed_state(G: AnalysisGraph) -> Dict:
    """ Create a dict corresponding to an observed state vector. """
    return {
        n[0]: {ind.name: ind.value for ind in n[1]["indicators"].values()}
        for n in G.nodes(data=True)
    }


class Sampler:
    """ MCMC sampler. """

    def __init__(self, G: AnalysisGraph):
        self.G = G
        self.A = None
        self.n_timesteps: int = None
        self.observed_state_sequence = None
        self.score: float = None
        self.candidate_score: float = None
        self.max_score: float = None
        self.delta_t = 1.0
        self.index_permutations = list(permutations(range(2 * len(G)), 2))
        self.s0 = G.construct_default_initial_state()

    def set_number_of_timesteps(self, n: int):
        self.n_timesteps = n

    def sample_observed_state(self, s: pd.Series) -> Dict:
        """ Sample an observed state given a latent state vector. """
        for n in self.G.nodes(data=True):
            for indicator in n[1]["indicators"].values():
                indicator.value = np.random.normal(
                    s[n[0]] * indicator.mean, indicator.stdev
                )

        return create_observed_state(self.G)

    def sample_from_prior(self):
        elements = self.G.get_latent_state_components()
        self.A = pd.DataFrame(
            np.identity(2 * len(self.G)), index=elements, columns=elements
        )
        for n in self.G.nodes:
            self.A[f"∂({n})/∂t"][n] = self.delta_t
        for e in self.G.edges(data=True):
            self.A[f"∂({e[0]})/∂t"][e[1]] = (
                e[2]["ConditionalProbability"].resample(1)[0][0] * self.delta_t
            )

    def sample_from_likelihood(self):
        self.observed_state_sequence = [
            self.sample_observed_state(s) for s in self.latent_state_sequence
        ]

    def set_latent_state_sequence(self):
        self.latent_state_sequence = ltake(
            self.n_timesteps,
            iterate(
                lambda s: pd.Series(self.A.values @ s.values, index=s.index),
                self.s0,
            ),
        )

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
        s0 = self.latent_state_sequence[0]
        _list = []
        for latent_state, observed_state in zip(
            self.latent_state_sequence, self.observed_state_sequence
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

    def sample_from_posterior(self, yield_sample=False):
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
            pass
        else:
            self.A.values[i][j] = original_value
