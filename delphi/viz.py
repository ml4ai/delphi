import networkx as nx
from .types import CausalAnalysisGraph

def to_agraph(cag:CausalAnalysisGraph):
    A = nx.nx_agraph.to_agraph(nx.DiGraph([(e[0].capitalize(), e[1].capitalize())
                                        for e in cag.edges()]))

    A.graph_attr['dpi']=300
    A.graph_attr['fontsize']=5
    A.graph_attr['overlap']='scale'
    A.graph_attr['rankdir'] = 'LR'
    A.edge_attr.update({'arrowsize': 0.5, 'color': '#650021'})
    A.node_attr.update({'shape': 'rectangle', 'color': '#650021',
                        'style':'rounded', 'fontname':'Gill Sans'})
    return A
