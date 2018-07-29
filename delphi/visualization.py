import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt

plt.style.use("ggplot")
import seaborn as sns
from networkx import DiGraph
from IPython.core.display import Image
from .export import to_agraph
from .AnalysisGraph import AnalysisGraph
from .utils import _insert_line_breaks

# ==========================================================================
# Visualization
# ==========================================================================


def visualize(G: DiGraph, *args, **kwargs):
    """ Visualize the analysis graph in a Jupyter notebook cell. """

    return Image(
        to_agraph(G, *args, **kwargs).draw(
            format="png", prog=kwargs.get("prog", "dot")
        ),
        retina=True,
    )
