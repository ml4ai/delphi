from .jupyter_tools import create_statement_inspection_table
from indra.statements import Influence
from itertools import chain
from .AnalysisGraph import AnalysisGraph

# ==========================================================================
# Inspection
# ==========================================================================

def _get_edge_sentences(G: AnalysisGraph, source: str, target: str) -> List[str]:
    """ Return the sentences that led to the construction of a specified edge.

    Args:
        source: The source of the edge.
        target: The target of the edge.
        cag: The analysis graph.
    """

    return chain.from_iterable(
        [
            [repr(e.text) for e in s.evidence]
            for s in G.edges[source, target]["InfluenceStatements"]
        ]
    )

def inspect_edge(G: AnalysisGraph, source: str, target: str):
    """ 'Drill down' into an edge in the analysis graph and inspect its
    provenance. This function prints the provenance."""
    return create_statement_inspection_table(
        G[source][target]["InfluenceStatements"]
    )


@property
def statements(G: AnalysisGraph) -> Iterable[Influence]:
    chainMap = compose(chain.from_iterable, map)
    sts = chainMap(
        lambda e: e[2]["InfluenceStatements"], G.edges(data=True)
    )
    return sorted(
        sts,
        key=lambda s: (
            s.subj.db_refs["UN"][0][0].split("/")[-1]
            + s.obj.db_refs["UN"][0][0].split("/")[-1]
        ),
    )

