from typing import Tuple, List
from .AnalysisGraph import AnalysisGraph
from .random_variables import LatentVar
from functools import singledispatch
from .execution import (
    default_update_function,
    emission_function,
    get_latent_state_components,
)
import pandas as pd
import numpy as np
from .program_analysis.ProgramAnalysisGraph import ProgramAnalysisGraph

# ==========================================================================
# Basic Modeling Interface (BMI)
# ==========================================================================


@singledispatch
def initialize():
    pass


def update_node(G: ProgramAnalysisGraph, n: str):
    """ Update the value of node n, recursively visiting its ancestors. """
    node = G.nodes[n]
    if node.get("update_fn") is not None and not node["visited"]:
        node["visited"] = True
        for p in G.predecessors(n):
            update_node(G, p)
        ivals = {i: G.nodes[i]["value"] for i in G.predecessors(n)}
        node["value"] = node["update_fn"](**ivals)


@initialize.register(ProgramAnalysisGraph)
def _(G: ProgramAnalysisGraph) -> ProgramAnalysisGraph:
    """ Initialize the value of nodes that don't have a predecessor in the
    CAG."""

    for n in G.nodes():
        if G.nodes[n].get("init_fn") is not None:
            G.nodes[n]["value"] = G.nodes[n]["init_fn"]()
    update(G)
    return G


@initialize.register(AnalysisGraph)
def _(G: AnalysisGraph, config_file: str) -> AnalysisGraph:
    """ Initialize the executable AnalysisGraph with a config file.

    Args:
        G
        config_file

    Returns:
        AnalysisGraph
    """
    G.s0 = pd.read_csv(
        config_file, index_col=0, header=None, error_bad_lines=False
    )[1]
    for n in G.nodes(data=True):
        n[1]["rv"] = LatentVar(n[0])
        n[1]["update_function"] = default_update_function
        node = n[1]["rv"]
        node.dataset = [G.s0[n[0]] for _ in range(G.res)]
        node.partial_t = G.s0[f"∂({n[0]})/∂t"]
        if n[1].get("indicators") is not None:
            for ind in n[1]["indicators"]:
                ind.dataset = np.ones(G.res) * ind.mean
    return G


@singledispatch
def update():
    pass


@update.register(ProgramAnalysisGraph)
def _(G: ProgramAnalysisGraph):

    for n in G.nodes():
        update_node(G, n)

    for n in G.nodes(data=True):
        n[1]["visited"] = False


@update.register(AnalysisGraph)
def _(G: AnalysisGraph) -> AnalysisGraph:
    """ Advance the model by one time step.

    Args:
        G

    Returns:
        AnalysisGraph
    """

    next_state = {}

    for n in G.nodes(data=True):
        next_state[n[0]] = n[1]["update_function"](G, n)

    for n in G.nodes(data=True):
        n[1]["rv"].dataset = next_state[n[0]]
        indicators = n[1].get("indicators")
        if (indicators is not None) and (indicators != []):
            ind = n[1]["indicators"][0]
            ind.dataset = [
                emission_function(x, ind.mean, ind.stdev)
                for x in n[1]["rv"].dataset
            ]

    G.t += G.Δt
    return G


def update_until(G: AnalysisGraph, t_final: float) -> AnalysisGraph:
    """ Updates the model to a particular time t_final """
    while G.t < t_final:
        update(G)

    return G


def finalize(G: AnalysisGraph):
    pass


# Model information


def get_component_name(G: AnalysisGraph) -> str:
    """ Return the name of the model. """
    return G.name


def get_input_var_names(G: AnalysisGraph) -> List[str]:
    """ Returns the input variable names """
    return get_latent_state_components(G)


def get_output_var_names(G: AnalysisGraph) -> List[str]:
    """ Returns the output variable names. """
    return get_latent_state_components(G)


def get_time_step(G: AnalysisGraph) -> float:
    """ Returns the time step size """
    return G.Δt


def get_time_units(G: AnalysisGraph) -> str:
    """ Returns the time unit. """
    return G.time_unit


def get_current_time(G: AnalysisGraph) -> float:
    """ Returns the current time in the execution of the model. """
    return G.t
