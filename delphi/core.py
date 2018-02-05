import pandas as pd
from delphi.utils import flatMap, lmap, compose, lfilter
from scipy.stats import gaussian_kde
from itertools import permutations
import numpy as np
from tqdm import trange
from typing import List, Tuple
from delphi.types import GroupBy, Delta
from indra.statements import Influence, Agent
from functools import partial
import pkg_resources
from networkx import DiGraph
from pandas import Series, DataFrame

# Location of the CLULab gradable adjectives data.
adjectiveData = pkg_resources.resource_filename('delphi',
                                                'data/adjectiveData.tsv')


def construct_CAG(factors: List[str], sts: List[Influence]) -> DiGraph:
    """ Constructs a networkx DiGraph that represents the causal analysis 
    graph.  """
    return DiGraph(lfilter(lambda t: len(t[2]['InfluenceStatements']) != 0,
        map(lambda p: (p[0], p[1], {'InfluenceStatements': filter(
                lambda s: p[0] == s.subj.name and p[1] == s.obj.name, sts)}),
            permutations(factors, 2))))



def add_conditional_probabilities(CAG: DiGraph) -> DiGraph:
    """ Add conditional probability information to the edges of the CAG. """

    # Create a pandas GroupBy object
    gb = pd.read_csv(adjectiveData, delim_whitespace=True).groupby('adjective')

    responses_kde = compose(gaussian_kde, get_respdevs)
    rs = flatMap(lambda g: responses_kde(g[1]).resample(5)[0].tolist(), gb)

    def get_adjective(delta: Delta) -> Optional[str]:
        """ Get the first adjective from subj_delta or obj_delta """

        if isinstance(delta['adjectives'], list):
            if len(delta['adjectives']) != 0:
                adj = delta['adjectives'][0]
            else:
                adj=None
        else:
            adj=delta['adjectives']

        if adj in gb.groups.keys():
            return adj
        else:
            return None

    def get_adjectives(statement: Influence) -> List[str]:
        """ Get the first adjective for the subj and obj in an INDRA statement.
        """
        return lmap(get_adjective, deltas(statement))

    for e in CAG.edges(data = True):
        simulableStatements = lfilter(
                lambda s: s.subj_delta['polarity'] is not None 
                and s.obj_delta['polarity'] is not None,
                e[2]['InfluenceStatements'])

        if len(simulableStatements) != 0:
            adjs=set(filter(lambda a: a is not None, 
                            flatMap(get_adjectives, simulableStatements)))
            adjectiveResponses = {a: get_respdevs(gb.get_group(a)) for a in adjs}

            def responses(adj) -> np.ndarray:
                if adj is None:
                    return rs
                else:
                    return adjectiveResponses[adj]

            def delta_responses(delta: Delta) -> np.array:
                return delta['polarity']*np.array(responses(get_adjective(delta)))

            response_tuples = lmap(lambda s: map(delta_responses, deltas(s)),
                    simulableStatements)

            rs_subj, rs_obj = list(*zip(response_tuples))[0]
            xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing='xy')

            if len({s.subj_delta['polarity']==s.obj_delta['polarity'] 
                    for s in simulableStatements})==1:
                xs2, ys2 = -xs1, -ys1
                thetas= np.append(
                    np.arctan2(ys1.flatten(), xs1.flatten()),
                    np.arctan2(ys2.flatten(), xs2.flatten())
                )
            else:
                thetas= np.arctan2(ys1.flatten(), xs1.flatten())

            e[2]['ConditionalProbability'] = gaussian_kde(thetas)
        else:
            e[2]['ConditionalProbability'] = None

    return CAG

def construct_transition_function(CAG: MultiDiGraph) -> DynamicBayesNet:
    factors = CAG.nodes()
    latent_state_components = flatMap(lambda a: (a, f'∂({a})/∂t'), factors)
    A = DataFrame(np.identity(len(latent_state_components)),
                      index = latent_state_components,
                      columns = latent_state_components)

    for c in latent_state_components[::2]:
        A[f'∂({a})/∂t'][f'{a}'] = 1
    for e in CAG.edges(data=True):
        if e[2]['ConditionalProbability'] is not None


    def transition_function(latent_state: np.ndarray) -> np.ndarray:
        return A @ latent_state

    return transition_function

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
    return s.subj_delta, s.obj_delta


def get_agents(statements: List[Influence]) -> List[str]:
    """
    Parameters
    ----------
    statements: 
        Influence statements

    Returns
    -------
    List[str]
        A list of agents.
    """

    factors = set(flatMap(lambda x: (x.subj.name, x.obj.name), statements))
    return flatMap(lambda a: (a, f'∂({a})/∂t'), sorted(factors))


def run_experiment(statements, s0, n_steps = 10, n_samples = 10, Δt = 1):
    gb = pd.read_csv(adjectiveData, delim_whitespace=True).groupby('adjective')
    get_kde = compose(gaussian_kde, get_respdevs)
    rs = flatMap(lambda g: get_kde(g[1]).resample(5)[0].tolist(), gb)
    simulableStatements = lfilter(lambda s: s.subj_delta['polarity'] != None 
            and s.obj_delta['polarity'] != None, statements)
    agents = get_agents(simulableStatements)

    def get_adjective(delta: Delta) -> str:
        if isinstance(delta['adjectives'], list):
            if len(delta['adjectives']) != 0:
                adj = delta['adjectives'][0]
            else:
                adj=None
        else:
            adj=delta['adjectives']
        if adj in gb.groups.keys():
            return adj
        else:
            return None

    def get_adjectives(s: Influence) -> List[str]:
        return lmap(get_adjective, deltas(s))

    adjs=set(filter(lambda a: a != None, flatMap(get_adjectives,
        simulableStatements)))

    adjectiveResponses = {a:get_respdevs(gb.get_group(a)) for a in adjs}

    def responses(adj) -> np.ndarray:
        if adj == None:
            return rs
        else:
            return adjectiveResponses[adj]

    def delta_responses(delta: Delta) -> np.array:
        return delta['polarity']*np.array(responses(get_adjective(delta)))

    def get_kde(a1, a2):
        # Get relevant statements
        sts = lfilter(lambda s: s.subj.name == a1 and s.obj.name== a2,
                simulableStatements)

        if len(sts) != 0:
            response_tuples = lmap(lambda s: lmap(delta_responses, deltas(s)), sts)
            rs_subj, rs_obj = list(*zip(response_tuples))[0]
            xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing='xy')

            if len({s.subj_delta['polarity']==s.obj_delta['polarity'] 
                    for s in sts})==1:
                xs2, ys2 = -xs1, -ys1
                thetas= np.append(
                    np.arctan2(ys1.flatten(), xs1.flatten()),
                    np.arctan2(ys2.flatten(), xs2.flatten())
                )
            else:
                thetas= np.arctan2(ys1.flatten(), xs1.flatten())

            return gaussian_kde(thetas)
        else:
            return None

    conditional_probabilities = {a1:{a2:get_kde(a1, a2) for a2 in agents} for a1 in agents}

    # Sample transition_matrix

    def sample_transition_matrix() -> np.ndarray:
        A = pd.DataFrame(np.identity(len(agents)), index = agents,
                         columns = agents)

        for a in agents[::2]:
            A[f'∂({a})/∂t'][f'{a}']=1

        for a1, a2 in permutations(agents, 2):
            if conditional_probabilities[a1][a2] != None:
                β= np.tan(conditional_probabilities[a1][a2].resample(50)[0][0])
                A[f'∂({a1})/∂t'][f'∂({a2})/∂t']=β*Δt

        return A.as_matrix()

    def sample_sequence() -> List:
        A = sample_transition_matrix()
        return ltake(n_steps, iterate(lambda x: A @ x, s0.values))

    return [sample_sequence(n_steps) for x in trange(n_samples)]


if __name__ == '__main__':
    statements = [Influence(
        Agent('X'),
        Agent('Y'),
        {'adjectives': [], 'polarity': None},
        {'adjectives': [], 'polarity': None},
    ),
    Influence(
        Agent('Y'),
        Agent('Z'),
        {'adjectives': [], 'polarity': None},
        {'adjectives': [], 'polarity': None},
    ),
    ]
    G =  constructCAG(['X', 'Y', 'Z'], statements)
    print(G.edges(data = True))

