from itertools import permutations, cycle
from typing import List, Tuple, Callable, Optional, Any, Dict, IO

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
from delphi.utils import flatMap, compose, iterate, ltake, exists

# Location of the CLULab gradable adjectives data.
adjectiveData = pkg_resources.resource_filename('delphi',
                                                'data/adjectiveData.tsv')


def construct_default_initial_state(s_index: List[str]) -> Series:
    return Series(dict(lzip(s_index, ltake(len(s_index), cycle([100, 0])))))


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    return (s.subj_delta, s.obj_delta)


def nameTuple(s: Influence) -> Tuple[str, str]:
    return (s.subj.name, s.obj.name)


def construct_CAG_skeleton(sts: List[Influence]) -> DiGraph:
    def makeEdgeTuple(
            p: Tuple[str, str]) -> Tuple[str, str, Dict[str, List[Influence]]]:

        return p[0], p[1], {'InfluenceStatements': lfilter(
                        lambda s: (p[0], p[1]) == nameTuple(s), sts)}

    return DiGraph(lfilter(
        lambda e: len(e[2]['InfluenceStatements']) != 0,
        map(makeEdgeTuple, permutations(set(flatMap(nameTuple, sts)), 2))))


def get_respdevs(gb: GroupBy) -> np.ndarray:
    return gb['respdev']


def isSimulable(s: Influence) -> bool:
    return all(map(exists, map(lambda x: x['polarity'], deltas(s))))


def constructConditionalPDF(gb: GroupBy, rs, e) -> gaussian_kde:

    simulableStatements = lfilter(isSimulable, e[2]['InfluenceStatements'])

    if not simulableStatements:
        return None

    else:

        # Make a adjective-response dict.

        def get_adjective(d: Delta) -> Optional[str]:
            """ Get the first adjective from subj_delta or obj_delta """

            if isinstance(d['adjectives'], list):
                if d['adjectives']:
                    adj = d['adjectives'][0]
                else:
                    adj = None
            else:
                adj = d['adjectives']

            return adj if adj in gb.groups.keys() else None

        adjectiveResponses = {a: get_respdevs(gb.get_group(a))
                for a in set(filter(exists, flatMap(
                    lambda s: lmap(get_adjective, deltas(s)),
                    simulableStatements)))}

        def responses(adj: Optional[str]) -> np.ndarray:
            return adjectiveResponses[adj] if exists(adj) else rs


        rs_subj, rs_obj = list(*zip(lmap(
            lambda s: map(
                lambda d: d['polarity'] * np.array(responses(get_adjective(d))),
                deltas(s)),
            simulableStatements)))[0]

        xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing='xy')

        if len(lmap(
            lambda s: s.subj_delta['polarity'] == s.obj_delta['polarity'],
            simulableStatements)) == 1:

            xs2, ys2 = -xs1, -ys1
            thetas = np.append(np.arctan2(ys1.flatten(), xs1.flatten()),
                               np.arctan2(ys2.flatten(), xs2.flatten()))
        else:
            thetas = np.arctan2(ys1.flatten(), xs1.flatten())

        return gaussian_kde(thetas)


def add_conditional_probabilities(CAG: DiGraph) -> DiGraph:
    # Create a pandas GroupBy object
    gb = read_csv(adjectiveData, delim_whitespace=True).groupby('adjective')
    rs = flatMap(lambda g: gaussian_kde(get_respdevs(g[1]))
                          .resample(20)[0].tolist(), gb)

    for e in CAG.edges(data=True):
        e[2]['ConditionalProbability'] = constructConditionalPDF(gb, rs, e)

    return CAG


def get_latent_state_components(CAG: DiGraph) -> List[str]:
    return flatMap(lambda a: (a, f'∂({a})/∂t'), CAG.nodes())


def initialize_transition_matrix(cs: List[str], Δt = 1) -> DataFrame:
    A = DataFrame(np.identity(len(cs)), cs, cs)
    for c in cs[::2]: A[f'∂({c})/∂t'][f'{c}'] = Δt
    return A


def sample_transition_matrix(CAG: DiGraph, Δt: float = 1.0) -> DataFrame:
    A = initialize_transition_matrix(get_latent_state_components(CAG))

    for e in CAG.edges(data=True):
        if 'ConditionalProbability' in e[2].keys():
            β = np.tan(e[2]['ConditionalProbability'].resample(1)[0][0])
            A[f'∂({e[0]})/∂t'][f'∂({e[1]})/∂t'] = β * Δt

    return A


def sample_sequence(CAG: DiGraph, s0: np.ndarray,
                    n_steps: int, Δt: float = 1.0) -> List[np.ndarray]:

    A = sample_transition_matrix(CAG, Δt).as_matrix()
    return ltake(n_steps, iterate(lambda s: A @ s, s0))


def sample_sequences(CAG: DiGraph, s0: Series, steps: int, samples: int,
                     Δt: float = 1.0) -> List[Series]:
    """ Sample a collection of sequences for a CAG """

    s0_array = s0.tolist()

    return [sample_sequence(CAG, s0_array, steps, Δt) for x in trange(samples)]


def construct_executable_model(sts: List[Influence]) -> DiGraph:
    return add_conditional_probabilities(construct_CAG_skeleton(sts))


def get_units(n: str) -> str:
    return "units"


def get_dtype(n: str) -> str:
    return "real"


def export_node(CAG: DiGraph, n) -> Dict:
    n[1]['name'] = n[0]
    n[1]['units'] = get_units(n[0])
    n[1]['dtype'] = get_dtype(n[0])
    n[1]['arguments'] = list(CAG.predecessors(n[0]))
    return n[1]


def export_model(CAG: DiGraph, f: IO[bytes]) -> None:
    lscs = get_latent_state_components(CAG)
    s0 = construct_default_initial_state(lscs)
    df = DataFrame(s0, columns = ['init_value'])
    df.to_csv('variables.csv', index_label='variable')
    # pickle.dump(CAG, f)


def load_model(f: IO[bytes]) -> DiGraph:
    return pickle.load(f)

def export_model_to_json(CAG: DiGraph):
    model = {
        'name' : 'Dynamic Bayes Net Model',
        'dateCreated' : str(datetime.datetime.now()),
        'variables' : lmap(partial(export_node, CAG), CAG.nodes(data=True))
        # 'edges' : list(dressed_CAG.edges(data = True)),
    }
    with open('cag.json', 'w') as f:
        json.dump(model, f, indent=2)
