import os
from pprint import pprint
import pandas as pd
from typing import List, Dict, Optional, Tuple
from indra.statements import Influence
from IPython.display import HTML, Code, Image
from future.utils import lmap
from .utils.indra import top_grounding_score
from .db import engine
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
import json
import pygments
import numpy as np
import subprocess as sp


def create_statement_inspection_table(sts: List[Influence]):
    """ Display an HTML representation of a table with INDRA statements to
    manually inspect for validity.

    Args:
        sts: A list of INDRA statements to be manually inspected for validity.
    """

    columns = [
        "un_groundings",
        "subj_polarity",
        "obj_polarity",
        "Sentence",
        "Source API",
    ]

    polarity_to_str = lambda x: "+" if x == 1 else "-" if x == -1 else "None"
    l = []
    for s in sts:
        subj_un_grounding = s.subj.db_refs["WM"][0][0].split("/")[-1]
        obj_un_grounding = s.obj.db_refs["WM"][0][0].split("/")[-1]
        subj_polarity = s.subj_delta["polarity"]
        obj_polarity = s.obj_delta["polarity"]
        subj_adjectives = s.subj_delta["adjectives"]
        for e in s.evidence:
            l.append(
                (
                    (subj_un_grounding, obj_un_grounding),
                    subj_polarity,
                    obj_polarity,
                    e.text,
                    e.source_api,
                )
            )

    df = pd.DataFrame(l, columns=columns)
    df = df.pivot_table(index=["un_groundings", "Source API", "Sentence"])

    def hover(hover_color="#ffff99"):
        return dict(
            selector="tr:hover",
            props=[("background-color", "%s" % hover_color)],
        )

    styles = [
        hover(),
        dict(props=[("font-size", "100%"), ("font-family", "Gill Sans")]),
    ]

    return df.style.set_table_styles(styles)


def print_full_edge_provenance(cag, source, target):
    for i, s in enumerate(cag.edges[source, target]["InfluenceStatements"]):
        print("Statement ", i)
        print("\t subj:", s.subj)
        print("\t obj:", s.obj)
        print("\t subj_delta:", s.subj_delta)
        print("\t obj_delta:", s.obj_delta)
        for e in s.evidence:
            print("\t obj_delta:", json.dumps(e.to_json(), indent=2))


def display(file: str):
    lexer = pygments.lexers.get_lexer_for_filename(file)

    with open(file, "r") as f:
        code = f.read()

    formatter = pygments.formatters.HtmlFormatter(
        linenos="inline", cssclass="pygments"
    )
    html_code = pygments.highlight(code, lexer, formatter)
    css = formatter.get_style_defs(".pygments")
    html = f"<style>{css}</style>{html_code}"

    return HTML(html)


def display_image(file: str):
    return Image(file, retina=True)


def get_python_shell():
    """Determine python shell

    get_python_shell() returns

    'shell' (started python on command line using "python")
    'ipython' (started ipython on command line using "ipython")
    'ipython-notebook' (e.g., running in Spyder or started with "ipython qtconsole")
    'jupyter-notebook' (running in a Jupyter notebook)

    See also https://stackoverflow.com/a/37661854
    """

    env = os.environ
    shell = "shell"
    program = os.path.basename(env["_"])

    if "jupyter-notebook" in program:
        shell = "jupyter-notebook"
    elif "JPY_PARENT_PID" in env or "ipython" in program:
        shell = "ipython"
        if "JPY_PARENT_PID" in env:
            shell = "ipython-notebook"

    return shell


def make_distplot(values, xlabel: str = ""):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    sns.distplot(values, ax=ax)


def get_percentile_based_conditional_forecast(
    forecast_var: str = "Production",
    conditioned_var: str = "Rainfall",
    crop="maize",
    percentile: float = 5,
):

    if forecast_var != "Production":
        raise NotImplementedError(
            "Currently, forecast_var must be set to 'Production'."
        )
    if conditioned_var != "Rainfall":
        raise NotImplementedError(
            "Currently, conditioned_var must be set to 'Rainfall'."
        )

    if crop not in ("maize", "sorghum"):
        raise NotImplementedError(
            "Currently, crop must be one of ['maize', 'sorghum']"
        )
    conditioned_var_values = list(
        engine.execute(
            f"select `{conditioned_var}` from dssat where `Crop` is '{crop}'"
        )
    )
    forecast_var_values = list(
        engine.execute(
            f"select `{forecast_var}` from dssat"
            f" where `{conditioned_var}` < "
            f"{np.percentile(conditioned_var_values, percentile)}"
        )
    )

    # TODO Make this more general
    unit = "tons"
    make_distplot(forecast_var_values, f"{forecast_var} of {crop} ({unit})")


def get_expected_distribution(
    indicator: str,
    state: Optional[str] = None,
    month_range: Optional[Tuple[int, int]] = None,
    unit: Optional[str] = None,
    method: str = "historical",
    table: str = "indicator",
):
    query_parts = [
        f"select * from '{table}' where `Variable` like '{indicator}'"
    ]

    if state is not None:
        query_parts.append(f"and `State` like '{state}'")
    if month_range is not None:
        query_parts.append(
            f"and `Month` between {month_range[0]} and {month_range[1]}"
        )

    query = " ".join(query_parts)
    results = list(engine.execute(query))
    values = [float(r["Value"]) for r in results]
    units = list(set([r["Unit"] for r in results]))
    if len(units) > 1:
        raise ValueError(
            "Multiple indicator variables found with the same name and "
            "different units. Please specify the units."
        )

    make_distplot(values, f"{indicator} ({units[0]})")


def print_commit_hash_message():
    commit_hash = sp.check_output(["git", "rev-parse", "HEAD"])
    print(
        f"This notebook has been rendered with commit {commit_hash[:-1]} of"
        " Delphi."
    )


def run_experiment(G, intervened_node, delta, n_timesteps: int,est = np.median,
        conf = 68,dampen = False):
    G.create_bmi_config_file()
    s0 = pd.read_csv(
        "bmi_config.txt", index_col=0, header=None, error_bad_lines=False
    )[1]
    s0.loc[f"∂({intervened_node})/∂t"] = delta
    s0.to_csv("bmi_config.txt", index_label="variable")
    G.initialize()
    indicator_values = {
        n[0]: {
            "name": list(n[1]["indicators"].keys())[0],
            "xs": [0 for _ in range(G.res)],
            "ys": list(list(n[1]["indicators"].values())[0].samples),
        }
        for n in G.nodes(data=True)
    }

    for t in range(1, n_timesteps + 1):
        G.update(dampen=dampen)
        for n in G.nodes(data=True):
            # TODO Fix this - make it more general (or justify it theoretically)
            ys = lmap(
                lambda x: x,
                list(n[1]["indicators"].values())[0].samples,
            )
            indicator_values[n[0]]["ys"].extend(ys)
            indicator_values[n[0]]["xs"].extend([t for _ in ys])

    for n in G.nodes(data=True):
        fig, ax = plt.subplots()
        sns.lineplot(
            indicator_values[n[0]]["xs"], indicator_values[n[0]]["ys"], ax=ax,
            ci=conf,
            estimator=est
        )
        ax.set_title(f"{indicator_values[n[0]]['name']} ({list(n[1]['indicators'].values())[0].unit})")
        ax.set_xlabel("time step number")


def perform_intervention(G, n0, partial_t_n0, n1, xlim=(0, 1)):
    G.construct_default_initial_state()
    G.create_bmi_config_file()
    s0 = pd.read_csv(
        "bmi_config.txt", index_col=0, header=None, error_bad_lines=False
    )[1]
    s0.loc[f"∂({n0})/∂t"] = partial_t_n0
    s0.to_csv("bmi_config.txt")
    G.initialize()
    vals = {}

    G.update()

    xmax = 2
    vals[n1] = [x for x in G.nodes[n1]["rv"].dataset if 0 < x < xmax]
    fig, ax = plt.subplots()
    ax.set_xlim(0, xmax)
    ax.set_xlabel(n1)
    ax.hist(vals[n1], density=True, bins=30)
    print(f"Standard deviation, σ = {np.std(vals[n1]):.2f}")
    plt.tight_layout()

def display(G, simplified_labels = True, label_depth=1, node_to_highlight=""):
    from pygraphviz import AGraph
    from IPython.core.display import Image

    temporary_image_filename = "tmp.png"
    try:
        G.to_png(temporary_image_filename, simplified_labels, label_depth, node_to_highlight)
        return Image(temporary_image_filename)
    finally:
        os.remove(temporary_image_filename)
