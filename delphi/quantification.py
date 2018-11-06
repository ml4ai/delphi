from typing import Optional
from .assembly import construct_concept_to_indicator_mapping, get_indicators
from .AnalysisGraph import AnalysisGraph
from .utils import get_data_from_url

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
        mapping_file
    """
    if mapping_file is None:
        url = "http://vision.cs.arizona.edu/adarsh/export/demos/data/concept_to_indicator_mapping.txt"
        mapping_file = get_data_from_url(url)

    mapping = construct_concept_to_indicator_mapping(n, mapping_file)

    for n in G.nodes(data=True):
        n[1]["indicators"] = get_indicators(
            n[0].lower().replace(" ", "_"), mapping
        )

    return G
