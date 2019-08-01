from delphi.cpp.AnalysisGraph import AnalysisGraph, InitialBeta
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
    G.map_concepts_to_indicators()
    G.print_indicators()
    print("\n")
    G.replace_indicator(
        "UN/events/human/human_migration",
        "Net migration",
        "New asylum seeking applicants",
        "UNHCR",
    )
    G.print_indicators()
    # Now we can specify how to initialize betas. Posible values are:
    # InitialBeta.ZERO
    # InitialBeta.ONE
    # InitialBeta.HALF
    # InitialBeta.MEAN
    # InitialBeta.RANDOM - A random value between [-1, 1]
    G.train_model(2015, 1, 2015, 12, 1000, 10000, initial_beta=InitialBeta.ZERO)
    preds = G.generate_prediction(2015, 1, 2016, 12)
    print(len(preds[0]))
    print(len(preds[1]))
    print(len(preds[1][0]))
    print(preds[0])
    predicted_point = preds[1][0]
    for ts in predicted_point:
        print(ts)
