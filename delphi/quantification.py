from typing import Optional
from .assembly import construct_concept_to_indicator_mapping

# ==========================================================================
# Quantification
# ==========================================================================


def map_concepts_to_indicators(
    G: self, n: int = 1, manual_mapping: Optional[dict] = None
) -> AnalysisGraph:
    """ Add indicators to the analysis graph.

    Args:
        G
        n
        manual_mapping
    """
    mapping = construct_concept_to_indicator_mapping(n=n)

    for n in G.nodes(data=True):
        n[1]["indicators"] = get_indicators(
            n[0].lower().replace(" ", "_"), mapping
        )
        if manual_mapping is not None:
            if n[0] in manual_mapping:
                n[1]["indicators"] = manual_mapping[n[0]]

    return G
