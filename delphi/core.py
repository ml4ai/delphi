import pandas as pd
from delphi.helpers import *
import scipy
from scipy.stats import gaussian_kde
from itertools import *
import numpy as np
from tqdm import trange
from typing import List, Dict, Tuple
from delphi.types import GroupBy
from indra.statements import Influence
import networkx as nx
import json
import pkg_resources
adjectiveData = pkg_resources.resource_filename('delphi', 'data/adjectiveData.tsv')

Delta = Dict[str, int]
np.set_printoptions(precision=4, linewidth=1000)
pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)
pd.set_option('precision',2)


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

def create_causal_analysis_graph(statements):
    agents = get_agents(statements)
    G = nx.MultiDiGraph() 

    for agent in agents[::2]:
        G.add_node(agent.capitalize(), simulable=False)
    for s in statements:
        subj, obj = s.subj.name.capitalize(), s.obj.name.capitalize()

        if s.subj_delta['polarity'] != None and s.obj_delta['polarity'] != None:
            G.nodes[subj]['simulable'] = True
            G.nodes[obj]['simulable'] = True

        key = G.add_edge(subj, obj,
                    subj_polarity = s.subj_delta['polarity'],
                    subj_adjectives = s.subj_delta['adjectives'],
                    obj_polarity = s.obj_delta['polarity'],
                    obj_adjectives = s.obj_delta['adjectives'],
                    linestyle='dotted'
                )
        if s.subj_delta['polarity'] != None and s.obj_delta['polarity'] != None:
            G[subj][obj][key]['linestyle']='solid'

    return G

def export_to_cytoscapejs(G: nx.DiGraph):
    """ Export networkx to format readable by CytoscapeJS """
    return {
            'nodes':[{'data':{'id':f'{n[0]}', 'simulable': n[1]['simulable']}} for n in G.nodes(data=True)],
            'edges':[
                {
                    'data':
                    {
                        'id'              : f'{e[0]}_{e[1]}',
                        'source'          : f'{e[0]}',
                        'target'          : f'{e[1]}',
                        'linestyle'       : f'{e[3]["linestyle"]}',
                        'subj_adjectives' : f'{e[3]["subj_adjectives"]}',
                        'subj_polarity'   : f'{e[3]["subj_polarity"]}',
                        'obj_adjectives' : f'{e[3]["obj_adjectives"]}',
                        'obj_polarity'   : f'{e[3]["obj_polarity"]}',
                        'simulable' : False if (e[3]['obj_polarity'] == None or
                            e[3]['subj_polarity'] == None) else True
                    }
                } 
                for e in G.edges(data=True, keys = True)
                ]
            }
 
def runExperiment(statements, s0, n_steps = 10, n_samples = 10, Δt = 1):
    # adjectiveData='data/adjectiveData.tsv' 
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

    def responses(adj):
        if adj == None:
            return rs
        else:
            return adjectiveResponses[adj]

    def delta_responses(Delta):
        return Delta['polarity']*np.array(responses(get_adjective(Delta)))

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
                thetas=np.append(
                    np.arctan2(ys1.flatten(), xs1.flatten()),
                    np.arctan2(ys2.flatten(), xs2.flatten())
                )
            else:
                thetas=np.arctan2(ys1.flatten(), xs1.flatten())

            return gaussian_kde(thetas)
        else:
            return None

    conditional_probabilities = {a1:{a2:get_kde(a1, a2) for a2 in agents} for a1 in agents}

    # # Sample transition_matrix

    def sample_transition_matrix():
        A = pd.DataFrame(np.identity(len(agents)), index = agents,
                         columns = agents)

        for a in agents[::2]:
            A[f'∂({a})/∂t'][f'{a}']=1

        for a1, a2 in permutations(agents, 2):
            if conditional_probabilities[a1][a2] != None:
                β=np.tan(conditional_probabilities[a1][a2].resample(50)[0][0])
                A[f'∂({a1})/∂t'][f'∂({a2})/∂t']=β*Δt

        return A.as_matrix()

    def sample_sequence(n_steps = n_steps):
        A = sample_transition_matrix()
        return ltake(n_steps, iterate(lambda x: A @ x, s0.values))

    return [sample_sequence(n_steps) for x in trange(n_samples)]


if __name__ == '__main__':
    create_animation(statements)
