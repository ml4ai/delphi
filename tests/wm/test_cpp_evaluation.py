from delphi.cpp.DelphiPython import AnalysisGraph, InitialBeta
from delphi.evaluation import pred_plot
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import seaborn as sns


def test_cpp_extensions_preds():
    statements = [
        (
            ("large", -1, "UN/entities/human/financial/economic/inflation"),
            ("small", 1, "UN/events/human/human_migration"),
        )
    ]
    G = AnalysisGraph.from_causal_fragments(statements)
    G.map_concepts_to_indicators()
    G["UN/events/human/human_migration"].replace_indicator(
        "Net migration", "New asylum seeking applicants", "UNHCR"
    )
    G.to_png()

    # Now we can specify how to initialize betas. Posible values are:
    # InitialBeta.ZERO
    # InitialBeta.ONE
    # InitialBeta.HALF
    # InitialBeta.MEAN
    # InitialBeta.RANDOM - A random value between [-1, 1]
    G.train_model(
        2015, 1, 2015, 12, 1000, 10000, initial_beta=InitialBeta.ZERO
    )
    preds = G.generate_prediction(2015, 1, 2016, 12)
    pred_plot(preds, "New asylum seeking applicants", save_as="pred_plot.pdf")


def test_delete_indicator():
    statements = [
        (
            ("large", -1, "UN/entities/human/financial/economic/inflation"),
            ("small", 1, "UN/events/human/human_migration"),
        )
    ]
    G = AnalysisGraph.from_causal_fragments(statements)
    print("\n")
    G.print_nodes()
    G.map_concepts_to_indicators()
    G.print_indicators()
    print("\n")
    G["UN/events/human/human_migration"].replace_indicator(
        "Net migration", "New asylum seeking applicants", "UNHCR"
    )
    G.print_indicators()
    print("\n")
    G.set_indicator(
        "UN/events/human/human_migration", "Net Migration", "MITRE12"
    )

    G.print_indicators()
    print("\n")
    G.delete_indicator(
        "UN/events/human/human_migration", "New asylum seeking applicants"
    )

    G.print_indicators()
    print("\n")

    G.set_indicator(
        "UN/events/human/human_migration",
        "New asylum seeking applicants",
        "UNHCR",
    )

    G.delete_all_indicators("UN/events/human/human_migration")

    G.print_indicators()
