from dataclasses import dataclass
import numpy as np
import pandas as pd
import random
from math import log
from scipy.stats import norm
from delphi.utils.fp import flatMap, ltake, iterate
from delphi.AnalysisGraph import AnalysisGraph
from typing import List, Dict
from itertools import permutations, product


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
        self.original_score = None

    def sample_initial_transition_matrix(self):
        for i, j in product(range(2 * len(self.G)), range(2 * len(self.G))):
            self.A.values[i][j] = np.random.normal()

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

    def update_log_prior(self) -> float:
        _list = [
            log(
                edge[2]["ConditionalProbability"].evaluate(
                    self.A[f"∂({edge[0]})/∂t"][edge[1]] / self.delta_t
                )
            )
            for edge in self.G.edges(data=True)
        ]

        self.log_prior = sum(_list)

    def update_log_likelihood(self) -> float:
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

        self.log_likelihood = sum(_list)

    def update_log_joint_probability(self):
        self.log_joint_probability = self.log_prior + self.log_likelihood

    def sample_from_proposal(self):
        # Choose the element of A to perturb
        self.source, self.target, self.edge_dict = random.choice(
            list(self.G.edges(data=True))
        )
        self.original_value = self.A[f"∂({self.source})/∂t"][self.target]
        self.A[f"∂({self.source})/∂t"][self.target] += np.random.normal(
            scale=0.1
        )

    def sample_from_posterior(self):
        self.sample_from_proposal()
        self.set_latent_state_sequence()
        self.update_log_prior()
        self.update_log_likelihood()

        candidate_log_joint_probability = self.log_prior + self.log_likelihood

        delta_log_joint_probability = (
            candidate_log_joint_probability - self.log_joint_probability
        )

        acceptance_probability = min(1, np.exp(delta_log_joint_probability))
        if acceptance_probability > np.random.rand():
            self.update_log_joint_probability()
        else:
            self.A[f"∂({self.source})/∂t"][self.target] = self.original_value
            self.set_latent_state_sequence()
            self.update_log_likelihood()
            self.update_log_prior()
            self.update_log_joint_probability()
