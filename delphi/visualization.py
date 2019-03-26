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
from pygraphviz import AGraph

# ==========================================================================
# Visualization
# ==========================================================================
