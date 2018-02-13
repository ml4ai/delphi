from itertools import permutations
from typing import List, Tuple, Callable, Optional

import numpy as np
import pkg_resources
from indra.statements import Influence
from networkx import DiGraph
from pandas import Series, DataFrame, read_csv
from scipy.stats import gaussian_kde
from tqdm import trange

from delphi.types import GroupBy, Delta
from delphi.utils import flatMap, lmap, compose, lfilter, iterate, ltake

# Location of the CLULab gradable adjectives data.
adjectiveData = pkg_resources.resource_filename('delphi',
                                                'data/adjectiveData.tsv')


def get_respdevs(gb: GroupBy) -> np.ndarray:
    """
    Parameters
    ----------
    gb:
        Groupby object

    Returns
    -------
    np.ndarray:
        An array of responses.
    """
    return gb['respdev']


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    """ Returns the delta dicts of an Influence statement as a 2-tuple """
    return s.subj_delta, s.obj_delta


def construct_CAG_skeleton(factors: List[str], sts: List[Influence]) -> DiGraph:
    """

    Parameters
    ----------
    factors:
        Factors that an analyst might query for.
    sts:
        List of available INDRA statements that might involve those factors.

    Returns
    -------
    DiGraph:
        A networkx DiGraph object that contains the link structure information.
        The nodes are the factors, and each edge contains the list of INDRA
        statements that involve the two nodes that it connects.

    """
    return DiGraph(lfilter(lambda t: len(t[2]['InfluenceStatements']) != 0,
                           map(lambda p: (
                           p[0], p[1], {'InfluenceStatements': lfilter(
                               lambda s: p[0] == s.subj.name and p[1] == s.obj.name,
                               sts)}),
                               permutations(factors, 2))))


def add_conditional_probabilities(CAG: DiGraph) -> DiGraph:
    """

    Parameters
    ----------
    CAG
        The causal analysis graph skeleton.

    Returns
    -------
    DiGraph
        The causal analysis graph, but now with conditional probability
        information encoded in the edges.

    """

    # Create a pandas GroupBy object
    gb = read_csv(adjectiveData, delim_whitespace=True).groupby('adjective')

    responses_kde = compose(gaussian_kde, get_respdevs)
    rs = flatMap(lambda g: responses_kde(g[1]).resample(5)[0].tolist(), gb)

    def get_adjective(delta: Delta) -> Optional[str]:
        """ Get the first adjective from subj_delta or obj_delta """

        if isinstance(delta['adjectives'], list):
            if len(delta['adjectives']) != 0:
                adj = delta['adjectives'][0]
            else:
                adj = None
        else:
            adj = delta['adjectives']

        if adj in gb.groups.keys():
            return adj
        else:
            return None

    def get_adjectives(statement: Influence) -> List[Optional[str]]:
        """
        Parameters
        ----------
        statement
            An INDRA Influence statement

        Returns
        -------
        List[Optional[str]]
            A list of adjectives (or None).

        """
        return lmap(get_adjective, deltas(statement))

    for e in CAG.edges(data=True):
        simulableStatements = lfilter(
            lambda s: s.subj_delta['polarity'] is not None
                      and s.obj_delta['polarity'] is not None,
            e[2]['InfluenceStatements'])

        if len(simulableStatements) != 0:

            # Collect adjectives
            adjs = set(filter(lambda a: a is not None,
                              flatMap(get_adjectives, simulableStatements)))

            # Make a adjective-response dict.
            adjectiveResponses = {a: get_respdevs(gb.get_group(a)) for a in
                                  adjs}

            def responses(adj) -> np.ndarray:
                if adj is None:
                    return rs
                else:
                    return adjectiveResponses[adj]

            def delta_responses(delta: Delta) -> np.array:
                return delta['polarity'] * np.array(
                    responses(get_adjective(delta)))

            response_tuples = lmap(lambda s: map(delta_responses, deltas(s)),
                                   simulableStatements)

            rs_subj, rs_obj = list(*zip(response_tuples))[0]
            xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing='xy')

            if len({s.subj_delta['polarity'] == s.obj_delta['polarity']
                    for s in simulableStatements}) == 1:
                xs2, ys2 = -xs1, -ys1
                thetas = np.append(
                    np.arctan2(ys1.flatten(), xs1.flatten()),
                    np.arctan2(ys2.flatten(), xs2.flatten())
                )
            else:
                thetas = np.arctan2(ys1.flatten(), xs1.flatten())

            e[2]['ConditionalProbability'] = gaussian_kde(thetas)
        else:
            e[2]['ConditionalProbability'] = None

    return CAG


def sample_transition_function(CAG: DiGraph) -> Callable[[Series], Series]:
    """

    Parameters
    ----------
    CAG:
        Causal analysis graph with conditional probabilities attached.

    Returns
    -------
        The transition function that evolves the latent state in a dynamic
        bayes net by one time step.

    """
    factors = CAG.nodes()
    latent_state_components = flatMap(lambda a: (a, f'∂({a})/∂t'), factors)
    A = DataFrame(np.identity(len(latent_state_components)),
                  index=latent_state_components,
                  columns=latent_state_components)

    # Initialize certain off-diagonal elements to represent discretized PDE
    # system update.
    for c in latent_state_components[::2]:
        A[f'∂({c})/∂t'][f'{c}'] = 1

    Δt = 1

    # Sample coefficients from conditional probability data in CAGs
    for e in CAG.edges(data=True):
        if e[2]['ConditionalProbability'] is not None:
            β = np.tan(e[2]['ConditionalProbability'].resample(50)[0][0])
            A[f'∂({e[0]})/∂t'][f'∂({e[1]})/∂t'] = β * Δt

    def transition_function(latent_state: Series) -> Series:
        return Series(A.as_matrix() @ latent_state.values, index = latent_state.index)

    return transition_function


def sample_sequences(statements, s0, n_steps, n_samples):
    """ Sample a collection of sequences for a CAG """
    factors = set(flatMap(lambda x: (x.subj.name, x.obj.name), statements))
    CAG = add_conditional_probabilities(
        construct_CAG_skeleton(factors, statements))

    def sample_sequence() -> List[Series]:
        transition_function = sample_transition_function(CAG)
        return ltake(n_steps, iterate(lambda x: transition_function(x), s0))

    return [sample_sequence() for x in trange(n_samples)]
