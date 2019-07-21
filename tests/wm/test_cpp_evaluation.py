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

    G.print_nodes()
    G.to_dot()
