import sys
import json
import pickle
from datetime import datetime
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from tqdm import trange, tqdm
from itertools import cycle, chain
from indra.statements import Influence, Concept
from networkx import DiGraph
from indra.sources import eidos
from pandas import Series, DataFrame, read_csv, isna, read_table

from delphi.types import GroupBy, Delta, AnalysisGraph, Node, Indicator
from delphi.utils import (
    flatMap,
    compose,
    iterate,
    ltake,
    exists,
    repeatfunc,
    take,
)

from typing import *


def construct_default_initial_state(s_index: List[str]) -> Series:
    return Series(ltake(len(s_index), cycle([1.0, 0.0])), s_index)


def get_latent_state_components(CAG: AnalysisGraph) -> List[str]:
    return flatMap(lambda a: (a, f"∂({a})/∂t"), CAG.nodes())


def initialize_transition_matrix(cs: List[str], Δt: float = 1) -> DataFrame:
    A = DataFrame(np.identity(len(cs)), cs, cs)
    for c in cs[::2]:
        A[f"∂({c})/∂t"][f"{c}"] = Δt
    return A


def sample_transition_matrix(
    CAG: AnalysisGraph, Δt: float = 1.0
) -> DataFrame:
    A = initialize_transition_matrix(get_latent_state_components(CAG))

    for e in CAG.edges(data=True):
        if "ConditionalProbability" in e[2].keys():
            β = np.tan(e[2]["ConditionalProbability"].resample(1)[0][0])
            A[f"∂({e[0]})/∂t"][f"∂({e[1]})/∂t"] = β * Δt

    return A


def sample_sequence_of_latent_states(
    CAG: AnalysisGraph, s0: np.ndarray, n_steps: int, Δt: float = 1.0
) -> List[np.ndarray]:

    A = sample_transition_matrix(CAG, Δt).values
    return take(n_steps, iterate(lambda s: A @ s, s0))


def sample_sequence_of_observed_states(CAG: AnalysisGraph, latent_states: List[np.ndarray]) -> List[np.ndarray]:
    return [get_observed_state(CAG, s) for s in latent_states]


def get_observed_state(CAG: AnalysisGraph, latent_state: object) -> object:
    latent_state_components = get_latent_state_components(CAG)
    observed_state = []
    for i, s in enumerate(latent_state_components):
        if i % 2 == 0:
            if CAG.node[s].get('indicators') is not None:
                for ind in CAG.node[s]['indicators']:
                    new_value = np.random.normal(latent_state[i]*ind.value, ind.stdev)
                    observed_state.append((ind.name, new_value))
            else:
                o = np.random.normal(latent_state[i], 0.1)
                observed_state.append((s, o))

    series = Series({k:v for k, v in observed_state})
    return series


def sample_sequences(
    CAG: AnalysisGraph,
    s0: Series,
    steps: int,
    samples: int,
    Δt: float = 1.0,
) -> List[Series]:
    """ Sample a collection of sequences for a CAG """

    s0 = s0.values[np.newaxis].T
    return sample_sequence_of_observed_states(
            CAG, ltake(samples, repeatfunc(sample_sequence_of_latent_states, CAG, s0, steps, Δt))
            )



def write_sequences_to_file(
    CAG: AnalysisGraph, seqs, output_filename: str
) -> None:
    with open(output_filename, "w") as f:
        f.write(
            ",".join(
                ["seq_no", "time_slice", *get_latent_state_components(CAG)[::2]]
            )
            + "\n"
        )
        for seq_no, seq in enumerate(seqs):
            for time_slice, latent_state in enumerate(seq):
                vs = ",".join([str(x) for x in latent_state[::2]])
                f.write(",".join([str(seq_no), str(time_slice), vs]) + "\n")
