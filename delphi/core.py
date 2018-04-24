from itertools import permutations, cycle
from typing import List, Tuple, Callable, Optional, Any, Dict

import pickle
import numpy as np
import pkg_resources
from indra.statements import Influence
from networkx import DiGraph
from pandas import Series, DataFrame, read_csv
from scipy.stats import gaussian_kde
from tqdm import trange

from functools import partial

import datetime
import json
from delphi.types import GroupBy, Delta
from future.utils import lmap, lfilter, lzip
from delphi.utils import flatMap, compose, iterate, ltake

# Location of the CLULab gradable adjectives data.
adjectiveData = pkg_resources.resource_filename('delphi',
                                                'data/adjectiveData.tsv')

def exists(x: Any) -> bool:
    return True if x is not None else False


def construct_default_initial_state(s_index: List[str]) -> Series:
    return compose(Series, dict, lzip)(s_index, ltake(len(s_index), cycle([100, 0])))


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    return (s.subj_delta, s.obj_delta)


def nameTuple(s: Influence) -> Tuple[str, str]:
    return (s.subj.name, s.obj.name)


def isSimulable(s: Influence) -> bool:
    return all(map(exists, map(lambda x: x['polarity'], deltas(s))))


def hasSamePolarity(s: Influence) -> bool:
    return s.subj_delta['polarity'] == s.obj_delta['polarity']


def checkMatch(p: Tuple[str, str], s: Influence) -> bool:
    return (p[0], p[1]) == nameTuple(s)


def makeEdge(sts: List[Influence], p: Tuple[str, str]) -> Tuple[str, str, Dict[str, List[Influence]]]:
    return p[0], p[1], {'InfluenceStatements': lfilter(partial(checkMatch, p), sts)}


def makeEdges(sts: List[Influence]) -> Tuple[str, str, Dict[str, List[Influence]]]:
    return map(partial(makeEdge, sts), permutations(set(flatMap(nameTuple, sts)), 2))


def isValidEdge(e: Tuple[str, str, Dict[str, List[Influence]]]) -> bool:
    return len(e[2]['InfluenceStatements']) != 0


def validEdges(es: List[Tuple[str, str, Dict[str, List[Influence]]]]) -> List[Tuple[str, str, Dict[str, List[Influence]]]]:
    return lfilter(isValidEdge, es)


def construct_CAG_skeleton(sts: List[Influence]) -> DiGraph:
    return compose(DiGraph, validEdges, makeEdges)(sts)


def get_respdevs(gb: GroupBy) -> np.ndarray:
    return gb['respdev']


def responses_kde(gb: np.ndarray) -> gaussian_kde:
    return gaussian_kde(get_respdevs(gb))


def get_adjective(gb: GroupBy, delta: Delta) -> Optional[str]:
    """ Get the first adjective from subj_delta or obj_delta """

    if isinstance(delta['adjectives'], list):
        if delta['adjectives']:
            adj = delta['adjectives'][0]
        else:
            adj = None
    else:
        adj = delta['adjectives']

    return adj if adj in gb.groups.keys() else None

def get_adjectives(gb: GroupBy, s: Influence) -> List[Optional[str]]:
    return lmap(partial(get_adjective, gb), deltas(s))


def responses(adj: Optional[str], rs) -> np.ndarray:
    return adjectiveResponses[adj] if exists(adj) else rs


def delta_responses(gb: GroupBy, rs, delta: Delta) -> np.array:
    return delta['polarity'] * np.array(responses(get_adjective(gb, delta), rs))


def constructConditionalPDF(gb: GroupBy, rs, e) -> gaussian_kde:

    simulableStatements = lfilter(isSimulable, e[2]['InfluenceStatements'])

    if not simulableStatements:
        return None
    else:
        # Collect adjectives
        adjs = set(filter(exists, flatMap(partial(get_adjectives, gb), simulableStatements)))

        # Make a adjective-response dict.
        adjectiveResponses = {a: get_respdevs(gb.get_group(a)) for a in adjs}

        response_tuples = lmap(lambda s: map(partial(delta_responses, gb, rs), deltas(s)), simulableStatements)

        rs_subj, rs_obj = list(*zip(response_tuples))[0]
        xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing='xy')


        if len(lmap(hasSamePolarity, simulableStatements)) == 1:
            xs2, ys2 = -xs1, -ys1
            thetas = np.append(
                np.arctan2(ys1.flatten(), xs1.flatten()),
                np.arctan2(ys2.flatten(), xs2.flatten())
            )
        else:
            thetas = np.arctan2(ys1.flatten(), xs1.flatten())

        return gaussian_kde(thetas)


def add_conditional_probabilities(CAG: DiGraph) -> DiGraph:
    # Create a pandas GroupBy object
    gb = read_csv(adjectiveData, delim_whitespace=True).groupby('adjective')
    rs = flatMap(lambda g: responses_kde(g[1]).resample(20)[0].tolist(), gb)

    for e in CAG.edges(data=True):
        e[2]['ConditionalProbability'] = constructConditionalPDF(gb, rs, e)

    return CAG


def get_latent_state_components(CAG: DiGraph) -> List[str]:
    return flatMap(lambda a: (a, f'∂({a})/∂t'), CAG.nodes())


def initialize_transition_matrix(cs: List[str], Δt = 1) -> DataFrame:
    A = DataFrame(np.identity(len(cs)), cs, cs)
    for c in cs[::2]: A[f'∂({c})/∂t'][f'{c}'] = Δt
    return A


def sample_transition_matrix(CAG: DiGraph, Δt = 1) -> DataFrame:
    A = initialize_transition_matrix(get_latent_state_components(CAG))

    for e in CAG.edges(data=True):
        if 'ConditionalProbability' in e[2].keys():
            β = np.tan(e[2]['ConditionalProbability'].resample(1)[0][0])
            A[f'∂({e[0]})/∂t'][f'∂({e[1]})/∂t'] = β * Δt

    return A


def sample_sequence(CAG: DiGraph, s0: np.ndarray, n_steps: int) -> List[np.ndarray]:
    A = sample_transition_matrix(CAG).as_matrix()
    return ltake(n_steps, iterate(lambda s: A @ s, s0))


def sample_sequences(CAG: DiGraph, s0: Series, n_steps: int, n_samples: int) -> List[Series]:
    """ Sample a collection of sequences for a CAG """

    s0_array = s0.tolist()

    return [sample_sequence(CAG, s0_array, n_steps) for x in trange(n_samples)]


def construct_executable_model(sts: List[Influence]):
    CAG = add_conditional_probabilities(construct_CAG_skeleton(sts))
    lscs = get_latent_state_components(CAG)
    s0 = construct_default_initial_state(lscs)
    return CAG


def get_units(n: str) -> str:
    return "units"

def get_dtype(n: str) -> str:
    return "real"

def export_node(CAG: DiGraph, n: str) -> Dict:
    n[1]['units'] = get_units(n[0])
    n[1]['dtype'] = get_dtype(n[0])


def export_model(CAG: DiGraph):
    with open('CAG.pkl', 'wb') as f:
        pickle.dump(CAG, f)

def load_model(filename: str) -> DiGraph:
    with open(filename, 'rb') as f:
        CAG = pickle.load(f)
    return CAG

# def export_model(CAG: DiGraph):
    # model = {
        # 'name' : 'Dynamic Bayes Net Model',
        # 'dateCreated' : str(datetime.datetime.now()),
        # 'variables' : lmap(export_node, CAG.nodes(data=True))
        # 'edges' : list(dressed_CAG.edges(data = True)),
    # }
