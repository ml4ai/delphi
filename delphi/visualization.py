import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
set_matplotlib_formats("retina")
plt.style.use("ggplot")
import seaborn as sns

# ==========================================================================
# Visualization
# ==========================================================================

def visualize(G: AnalysisGraph, *args, **kwargs):
    """ Visualize the analysis graph in a Jupyter notebook cell. """
    from IPython.core.display import Image

    return Image(
        to_agraph(G: AnalysisGraph, *args, **kwargs).draw(
            format="png", prog=kwargs.get("prog", "dot")
        ),
        retina=True,
    )

def plot_distribution_of_latent_variable(
    G: AnalysisGraph, latent_variable, ax, xlim=None, **kwargs
):
    displayName = kwargs.get(
        "displayName", _insert_line_breaks(latent_variable, 30)
    )
    vals = [s[latent_variable] for s in G.latent_state.dataset]
    if xlim is not None:
        ax.set_xlim(*xlim)
        vals = [v for v in vals if ((v > xlim[0]) and (v < xlim[1]))]
    sns.distplot(vals, ax=ax, kde=kwargs.get("kde", True), norm_hist=True)
    ax.set_xlabel(displayName)
    ax.set_ylabel(_insert_line_breaks(f"p({displayName})"))

def plot_distribution_of_observed_variable(
    G: AnalysisGraph, observed_variable, ax, xlim=None, **kwargs
):
    displayName = kwargs.get(
        "displayName", _insert_line_breaks(observed_variable, 30)
    )

    vals = [s[observed_variable] for s in G.observed_state.dataset]
    if xlim is not None:
        ax.set_xlim(*xlim)
        vals = [v for v in vals if (v > xlim[0]) and (v < xlim[1])]
    plt.style.use("ggplot")
    sns.distplot(vals, ax=ax, kde=kwargs.get("kde", True), norm_hist=True)
    ax.set_xlabel(displayName)
    ax.set_ylabel(_insert_line_breaks(f"p({displayName})"))
