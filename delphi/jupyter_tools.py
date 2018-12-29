import os
from pprint import pprint
import pandas as pd
from typing import List, Dict, Optional
from indra.statements import Influence
from IPython.display import HTML, Code, Image
from delphi.utils.indra import top_grounding_score
import json
import pygments


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
