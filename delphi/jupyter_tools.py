import pandas as pd
from typing import List
from indra.statements import Influence
from IPython.display import Image, HTML
from .viz import to_agraph
from .types import AnalysisGraph
from .assembly import top_grounding_score
import json

def create_statement_inspection_table(sts: List[Influence]):
    """ Display an HTML representation of a table with INDRA statements to
    manually inspect for validity.

    Args:
        sts: A list of INDRA statements to be manually inspected for validity.
    """

    columns = [ 'subj text', 'Canonical name (subj)', 'UN (subj)', 'obj text',
                'Canonical Name (obj)', 'UN (obj)', 'Sentence', ]

    df = pd.DataFrame([(s[1].subj.db_refs['TEXT'].replace('\n',' '),
                        s[1].subj,
                        (s[1].subj.db_refs['UN'][0][0].split('/')[-1],
                        f"{top_grounding_score(s[1].subj):.2f}"),
                        s[1].obj.db_refs['TEXT'].replace('\n',' '),
                        s[1].obj,
                        (s[1].obj.db_refs['UN'][0][0].split('/')[-1],
                        f"{top_grounding_score(s[1].obj):.2f}"),
                        s[1].evidence[0].text.replace('\n',' ')
                    ) for s in list(enumerate(sts))
            ], columns=columns)

    return HTML(df.to_html())


def visualize(cag: AnalysisGraph, *args, **kwargs):
    """ Visualize the analysis graph in a Jupyter notebook cell. """

    return Image(to_agraph(cag, *args, **kwargs).draw(format='png',
        prog=kwargs.get('prog', 'dot')))


def inspect_edge(source, target, cag: AnalysisGraph, **kwargs):
    """ 'Drill down' into an edge in the analysis graph and inspect its
    provenance. This function prints the provenance."""

    print_full_edge_provenance(source, target, cag)


def get_edge_sentences(source: str, target: str, cag: AnalysisGraph) -> List[str]:
    """ Return the sentences that led to the construction of a specified edge.

    Args:
        source: The source of the edge.
        target: The target of the edge.
        cag: The analysis graph.
    """
    return [[repr(e.text) for e in s.evidence]
            for s in cag.edges[source, target]['InfluenceStatements']]


def print_full_edge_provenance(source, target, cag):
    for i, s in enumerate(cag.edges[source, target]['InfluenceStatements']):
        print('Statement ',i)
        print('\t subj:', s.subj)
        print('\t obj:', s.obj)
        print('\t subj_delta:', s.subj_delta)
        print('\t obj_delta:', s.obj_delta)
        for e in s.evidence:
            print('\t obj_delta:', json.dumps(e.to_json(), indent=2))


