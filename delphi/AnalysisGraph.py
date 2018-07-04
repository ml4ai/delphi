import networkx as nx
from .assembly import make_cag_skeleton

class AnalysisGraph(nx.DiGraph):
    def to_agraph(self, *args, **kwargs):
        A = nx.nx_agraph.to_agraph(nx.DiGraph([(e[0].capitalize(), e[1].capitalize())
                                            for e in cag.edges()]))

        A.graph_attr['dpi']=300
        A.graph_attr['fontsize']=20
        A.graph_attr['overlap']='scale'
        A.graph_attr['rankdir'] = 'LR'
        A.edge_attr.update({'arrowsize': 0.5, 'color': '#650021'})
        A.node_attr.update({'shape': 'rectangle', 'color': '#650021',
                            'style':'rounded', 'fontname':'Gill Sans'})
        if kwargs.get('indicators') is not None:
            for n in cag.nodes(data = True):
                if n[1].get('indicators') is not None:
                    for ind in n[1]['indicators']:
                        node_label = _insert_line_breaks(ind.name)
                        A.add_node(node_label, style='rounded, filled',
                                fillcolor = 'lightblue')
                        A.add_edge(n[0].capitalize(), node_label)
        if kwargs.get('indicator_values'):
            for n in cag.nodes(data = True):
                if n[1].get('indicators') is not None:
                    for ind in n[1]['indicators']:
                        if ind.value is not None:
                            node_label = str(ind.value)
                            A.add_node(node_label, style='rounded, filled',
                                    fillcolor = 'white', color = "#fffff")
                            A.add_edge(_insert_line_breaks(ind.name), node_label)

        if kwargs.get('nodes_to_highlight') is not None:
            for n in kwargs['nodes_to_highlight']:
                A.add_node(n.capitalize(), fontcolor = 'royalblue')

        return A
