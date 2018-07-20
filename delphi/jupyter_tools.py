import pandas as pd
from typing import List, Dict, Optional
from indra.statements import Influence
from IPython.display import HTML, Code
from .assembly import top_grounding_score
import json
import pygments

def create_statement_inspection_table(sts: List[Influence]):
    """ Display an HTML representation of a table with INDRA statements to
    manually inspect for validity.

    Args:
        sts: A list of INDRA statements to be manually inspected for validity.
    """

    columns = [
        'subj',
        "UN (subj)",
        'subj_polarity',
        'subj_adjectives',
        'obj',
        "UN (obj)",
        'obj_polarity',
        'obj_adjectives',
        "Sentence",
    ]

    to_str = lambda x: '+' if x == 1 else '-' if x == -1 else 'None'
    df = pd.DataFrame(
        [
            (
                s[1].subj,
                (
                    s[1].subj.db_refs["UN"][0][0].split("/")[-1],
                    f"{top_grounding_score(s[1].subj):.2f}",
                ),
                to_str(s[1].subj_delta.get('polarity')),
                to_str(s[1].subj_delta.get('adjectives')),
                s[1].obj,
                (
                    s[1].obj.db_refs["UN"][0][0].split("/")[-1],
                    f"{top_grounding_score(s[1].obj):.2f}",
                ),
                to_str(s[1].obj_delta.get('polarity')),
                to_str(s[1].obj_delta.get('adjectives')),
                s[1].evidence[0].text.replace("\n", " "),
            )
            for s in list(enumerate(sts))
        ],
        columns=columns,
    )

    return HTML(df.to_html())


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

    with open(file, 'r') as f:
        code = f.read()

    formatter = pygments.formatters.HtmlFormatter(linenos='inline')
    html = pygments.highlight(code, lexer, formatter)

    return HTML(html)
