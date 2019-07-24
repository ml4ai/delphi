from delphi.cpp.AnalysisGraph import AnalysisGraph
import pytest
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import seaborn as sns


def test_cpp_extensions():
    statements = [
        (
            ("large", -1, "UN/entities/human/financial/economic/inflation"),
            ("small", 1, "UN/events/human/human_migration"),
        )
    ]
    G = AnalysisGraph.from_statements(statements)
    print("\n")
    G.print_nodes()
    G.map_concepts_to_indicators(1)
    G.print_indicators()
    print("\n")
    G.set_indicator(
        "UN/events/human/human_migration",
        "New asylum seeking applicants",
        "UNHCR",
        True,
        0,
    )
    G.print_indicators()