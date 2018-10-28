import matplotlib

from matplotlib import pyplot as plt

plt.style.use("ggplot")
import seaborn as sns
from networkx import DiGraph
from IPython.core.display import Image
from .export import to_agraph
from .AnalysisGraph import AnalysisGraph
from .utils.misc import _insert_line_breaks
from functools import singledispatch
from .program_analysis.ProgramAnalysisGraph import ProgramAnalysisGraph
from pygraphviz import AGraph

# ==========================================================================
# Visualization
# ==========================================================================

@singledispatch
def visualize():
    pass


@visualize.register(AnalysisGraph)
def _(G: AnalysisGraph, *args, **kwargs):
    """ Visualize the analysis graph in a Jupyter notebook cell. """

    return Image(
        to_agraph(G, *args, **kwargs).draw(
            format="png", prog=kwargs.get("prog", "dot")
        ),
        retina=True,
    )


@visualize.register(ProgramAnalysisGraph)
def _(G: ProgramAnalysisGraph, show_values=False):
    """ Exports AnalysisGraph to pygraphviz AGraph

    Args:
        args
        kwargs

    Returns:
        AGraph
    """

    A = AGraph(directed=True)
    A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo"})
    A.node_attr.update(
        {
            "shape": "rectangle",
            "color": "#650021",
            "style": "rounded",
            "fontname": "Gill Sans",
        }
    )

    color_str = "#650021"

    for n in G.nodes():
        A.add_node(n, label=n)

    for e in G.edges(data=True):
        A.add_edge(e[0], e[1], color=color_str, arrowsize=0.5)

    if show_values:
        for n in A.nodes():
            value = str(G.nodes[n]["value"])
            n.attr["label"] = n.attr["label"] + f": {value:.4}"

    # Drawing indicator variables

    return Image(A.draw(format="png", prog="dot"), retina=True)
