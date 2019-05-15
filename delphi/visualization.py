import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt

plt.style.use("ggplot")
import seaborn as sns
from networkx import DiGraph
from IPython.core.display import Image
from .AnalysisGraph import AnalysisGraph
from .utils.misc import _insert_line_breaks
from functools import singledispatch
from .AnalysisGraph import AnalysisGraph

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
        G.to_agraph(*args, **kwargs).draw(
            format="png", prog=kwargs.get("prog", kwargs.get("layout", "dot"))
        ),
        retina=True,
    )
