import os
from pprint import pprint
import pandas as pd
from typing import List, Dict, Optional
from indra.statements import Influence
from IPython.display import HTML, Code, Image
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
        subj_un_grounding = s.subj.db_refs["UN"][0][0].split("/")[-1]
        obj_un_grounding = s.obj.db_refs["UN"][0][0].split("/")[-1]
        subj_polarity = s.subj_delta["polarity"]
        obj_polarity = s.obj_delta["polarity"]
        subj_adjectives = s.subj_delta["adjectives"]
        for e in s.evidence:
            # sent = [e.text for e in s.evidence]
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


def get_expected_value(quantity: str, state: str, method="historical", **kwargs):

    if method == "historical":
        if quantity == "rainfall":
            units = "mm"
            results = engine.execute(
                f"select {quantity.capitalize()} from dssat where `Crop` like 'maize'"
                f" and `State` like '{state}'"
            )
            values = [r[0] for r in results]
            print(
                f"Based on historical data, the mean expected {quantity} for the lean "
                f"season for the state of {state} is {np.mean(values):.2f} {units} "
                f"with a standard deviation of {np.std(values):.2f} {units}."
            )
            xlabel = f"Historical distribution of {quantity} for {state} ({units})"
        elif quantity == "production":
            units = "tonnes"
            crop = kwargs.get('crop')
            if crop not in ("maize", "sorghum"):
                raise ValueError("The 'crop' keyword argument must be one of "
                                 "the following:  ('maize', 'sorghum')")
            results = engine.execute(
                f"select {quantity.capitalize()} from dssat where `Crop` like "
                f"'{crop}' and `State` like '{state}'"
            )
            values = [r[0] for r in results]
            print(
                f"Based on historical data, the mean expected {crop} {quantity} "
                f"for the state of {state} is {np.mean(values):.2f} {units} "
                f"with a standard deviation of {np.std(values):.2f} {units}."
            )
            xlabel = f"Historical distribution of {crop} \n{quantity} for {state} ({units})"
        else:
            raise ValueError("'quantity' must be one of the following: "
                             "('rainfall', 'production')")
    else:
        raise NotImplementedError("Estimates can currently only be obtained "
                "using historical data - please set method='historical'. "
                "In future versions, predictions will be obtained from domain "
                "model simulations.")

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    sns.distplot(values, ax=ax)


def print_commit_hash_message():
    commit_hash = sp.check_output(["git", "rev-parse", "HEAD"])
    print(f"This notebook has been rendered with commit {commit_hash[:-1]} of"
            " Delphi.")

def run_experiment(G, n0, partial_t_n0, n1, xlim = (0,1)):
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
    ax.set_xlim(0,xmax)
    ax.set_xlabel(n1)
    ax.hist(vals[n1], density=True, bins=30)
    print(f"Standard deviation, σ = {np.std(vals[n1]):.2f}")
    plt.tight_layout()
