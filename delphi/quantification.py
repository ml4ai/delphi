from typing import Optional
from .assembly import construct_concept_to_indicator_mapping, get_indicators
from .AnalysisGraph import AnalysisGraph

# ==========================================================================
# Quantification
# ==========================================================================


def map_concepts_to_indicators(
        G: AnalysisGraph, n: int = 1, mapping_file: Optional[str] = None
) -> AnalysisGraph:
    """ Add indicators to the analysis graph.

    Args:
        G
        n
        manual_mapping
    """
    mapping = construct_concept_to_indicator_mapping(n, mapping_file)

    for n in G.nodes(data=True):
        n[1]["indicators"] = get_indicators(
            n[0].lower().replace(" ", "_"), mapping
        )

    return G
