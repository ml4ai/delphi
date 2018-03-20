from itertools import permutations
from typing import List, Tuple, Callable, Optional

import numpy as np
import pkg_resources
from indra.statements import Influence
from networkx import DiGraph
from pandas import Series, DataFrame, read_csv
from scipy.stats import gaussian_kde
from tqdm import trange

import datetime
import json
from delphi.types import GroupBy, Delta
from future.utils import lmap, lfilter
from delphi.utils import flatMap, compose, iterate, ltake

# Location of the CLULab gradable adjectives data.
adjectiveData = pkg_resources.resource_filename('delphi',
                                                'data/adjectiveData.tsv')

def exists(x):
    return True if x is not None else False


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


def construct_CAG_skeleton(sts: List[Influence]) -> DiGraph:
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
    nameTuple = lambda stmt: (stmt.subj.name, stmt.obj.name)
    checkMatch = lambda perm: lambda stmt: (perm[0], perm[1]) == nameTuple(stmt)
    makeEdge = lambda perm: (perm[0], perm[1],
            {'InfluenceStatements': lfilter(checkMatch(perm), sts)})
    edges = map(makeEdge, permutations(set(flatMap(nameTuple, sts)), 2))
    validEdges = lfilter(lambda t: len(t[2]['InfluenceStatements']) != 0, edges)

    return DiGraph(validEdges)


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
            if delta['adjectives']:
                adj = delta['adjectives'][0]
            else:
                adj = None
        else:
            adj = delta['adjectives']

        return adj if adj in gb.groups.keys() else None

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

    isSimulable = lambda s: all(map(exists, map(lambda x: x['polarity'], deltas(s))))

    def constructConditionalPDF(simulableStatements):
        # Collect adjectives
        adjs = set(filter(exists, flatMap(get_adjectives, simulableStatements)))

        # Make a adjective-response dict.
        adjectiveResponses = {a: get_respdevs(gb.get_group(a)) for a in adjs}
        responses = lambda adj: adjectiveResponses[adj] if exists(adj) else rs

        def delta_responses(delta: Delta) -> np.array:
            return delta['polarity'] * np.array(responses(get_adjective(delta)))

        response_tuples = lmap(lambda s: map(delta_responses, deltas(s)),
                               simulableStatements)

        rs_subj, rs_obj = list(*zip(response_tuples))[0]
        xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing='xy')

        hasSamePolarity = \
            lambda s: s.subj_delta['polarity'] == s.obj_delta['polarity']

        if len(lmap(hasSamePolarity, simulableStatements)) == 1:
            xs2, ys2 = -xs1, -ys1
            thetas = np.append(
                np.arctan2(ys1.flatten(), xs1.flatten()),
                np.arctan2(ys2.flatten(), xs2.flatten())
            )
        else:
            thetas = np.arctan2(ys1.flatten(), xs1.flatten())

        return gaussian_kde(thetas)

    def attachConditionalProbability(e):
        simulableStatements = lfilter(isSimulable, e[2]['InfluenceStatements'])

        if simulableStatements:
            e[2]['ConditionalProbability'] = \
                constructConditionalPDF(simulableStatements)
        else:
            e[2]['ConditionalProbability'] = None

    for e in CAG.edges(data=True):
        attachConditionalProbability(e)

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
    latent_state_components = flatMap(lambda a: (a, f'∂({a})/∂t'), CAG.nodes())
    A = DataFrame(np.identity(len(latent_state_components)),
                  latent_state_components, latent_state_components)

    # Initialize certain off-diagonal elements to represent discretized PDE
    # system update.
    for c in latent_state_components[::2]:
        A[f'∂({c})/∂t'][f'{c}'] = 1

    Δt = 1

    # Sample coefficients from conditional probability data in CAGs
    for e in CAG.edges(data=True):
        if exists(e[2]['ConditionalProbability']):
            β = np.tan(e[2]['ConditionalProbability'].resample(50)[0][0])
            A[f'∂({e[0]})/∂t'][f'∂({e[1]})/∂t'] = β * Δt

    def transition_function(latent_state: Series) -> Series:
        return Series(A.as_matrix() @ latent_state.values, latent_state.index)

    return transition_function



def add_parents(CAG: DiGraph) -> DiGraph:
    for n in CAG.nodes(data = True):
        n[1]['arguments'] = list(CAG.predecessors(n[0]))
        n[1]['function'] = {
            "operation" : "cpd",
                }
    return CAG


def add_types(CAG: DiGraph) -> DiGraph:
    for n in CAG.nodes(data = True):
        n[1]['type'] = 'Type'
    return CAG

def add_units(CAG: DiGraph) -> DiGraph:
    for n in CAG.nodes(data = True):
        n[1]['units'] = 'Units'
        n[1]['type'] = 'Type'
    return CAG


def makeJSONSerializable(CAG: DiGraph) -> DiGraph:
    for e in CAG.edges(data = True):
        e[2]['InfluenceStatements'] = \
                lmap(lambda x: x.to_json(), e[2]['InfluenceStatements'])
        del e[2]['ConditionalProbability']
    return CAG


def create_dressed_CAG(statements: List[Influence]) -> List[Influence]:
    """ Attach conditional probability information to CAG edges """
    return compose(
            makeJSONSerializable,
            add_parents,
            add_units,
            add_types,
            add_conditional_probabilities,
            construct_CAG_skeleton,
            )(statements)


def sample_sequences(statements: List[Influence],
        s0: Series, n_steps: int, n_samples: int) -> List[Series]:
    """ Sample a collection of sequences for a CAG """

    dressed_CAG = create_dressed_CAG(statements)
    def sample_sequence() -> List[Series]:
        transition_function = sample_transition_function(dressed_CAG)
        return ltake(n_steps, iterate(lambda x: transition_function(x), s0))

    return [sample_sequence() for x in trange(n_samples)]


def export_model(statements):
    dressed_CAG = create_dressed_CAG(statements)

    model = {
        'name' : 'Dynamic Bayes Net Model',
        'dateCreated' : str(datetime.datetime.now()),
        'variables' : list(dressed_CAG.nodes(data = True)),
        'edges' : list(dressed_CAG.edges(data = True)),
    }

    with open('model.json', 'w') as f:
        json.dump(model, f, indent=2)
