from .AnalysisGraph import AnalysisGraph
from itertools import cycle
from typing import List, Tuple
import numpy as np
import pandas as pd
from .utils import flatMap, ltake


# ==========================================================================
# Execution
# ==========================================================================

def default_update_function(G: AnalysisGraph, n: Tuple[str, dict]) -> List[float]:
    rv = n[1]["rv"]
    return [
            rv.dataset[i]
            + (
                rv.partial_t
                + sum(
                    G[p][n[0]]["betas"][i] * G.nodes[p]["rv"].partial_t
                    for p in G.pred[n[0]]
                )
            )
            * G.Δt
            for i in range(G.res)
        ]


def emission_function(s_i, mu_ij, sigma_ij):
    return np.random.normal(s_i*mu_ij, sigma_ij)


def get_latent_state_components(G) -> List[str]:
    return flatMap(lambda a: (a, f"∂({a})/∂t"), G.nodes())

def _write_latent_state(G, f):
    for i, s in enumerate(G.latent_state.dataset):
        f.write(str(i) + "," + str(G.get_current_time()) + ",")
        f.write(",".join([str(v) for v in s.values[::2]]) + "\n")

def _write_sequences_to_file(G: AnalysisGraph, seqs, output_filename: str) -> None:
    with open(output_filename, "w") as f:
        f.write(
            ",".join(
                [
                    "seq_no",
                    "time_slice",
                    *get_latent_state_components(G)[::2],
                ]
            )
            + "\n"
        )
        for seq_no, seq in enumerate(seqs):
            for time_slice, latent_state in enumerate(seq):
                vs = ",".join([str(x) for x in latent_state[::2]])
                f.write(",".join([str(seq_no), str(time_slice), vs]) + "\n")

def construct_default_initial_state(G) -> pd.Series:
    comps = get_latent_state_components(G)
    return pd.Series(
        ltake(len(comps), cycle([1.0, 0.0])), comps
    )

