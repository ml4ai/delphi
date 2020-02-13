import numpy as np
import pytest
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from delphi.cpp.DelphiPython import AnalysisGraph, Indicator


food_security = "wm/concept/causal_factor/condition/food_security"
inflation = "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/inflation"
tension = "wm/concept/causal_factor/condition/tension"
displacement = "wm/concept/indicator_and_reported_property/conflict/population_displacement"
crop_production = (
    "wm/concept/indicator_and_reported_property/agriculture/Crop_Production"
)


def test_cpp_extensions():
    G = AnalysisGraph.from_indra_statements_json_file("tests/data/indra_statements_format.json")


def test_simple_path_construction():
    G = AnalysisGraph.from_indra_statements_json_file("tests/data/indra_statements_format.json")
    G.add_node("c0")
    G.add_node("c1")
    G.add_node("c2")

    print("Nodes of the graph:")
    G.print_nodes()

    G.add_edge((("", 1, "c0"), ("", 1, "c1")))
    G.add_edge((("", 1, "c1"), ("", 1, "c2")))
    G.add_edge((("", 1, "c0"), ("", 1, "c2")))
    G.add_edge(
        (("", 1, "c3"), ("", 1, "c1"))
    )  # Creates a loop 1 -> 2 -> 3 -> 1

    print("Edges of the graph:")
    G.print_edges()

    G.find_all_paths()
    G.print_all_paths()

    G.print_cells_affected_by_beta(0, 1)
    G.print_cells_affected_by_beta(1, 2)

    G2 = AnalysisGraph.from_indra_statements_json_file(
        "tests/data/indra_statements_format.json"
    )


def test_inference():
    causal_fragments = [(("small", 1, tension), ("large", -1, food_security))]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)

    G.print_nodes()

    print("\nName to vertex ID map entries")
    G.print_name_to_vertex()


def test_remove_node():
    causal_fragments = [(("small", 1, tension), ("large", -1, food_security))]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()
    G.print_nodes()

    print("\nRemoving an invalid concept")
    with pytest.raises(IndexError):
        G.remove_node(concept="invalid")

    print("\nRemoving a valid concept")
    G.remove_node(concept=tension)
    G.print_nodes()


def test_remove_nodes():
    causal_fragments = [(("small", 1, tension), ("large", -1, food_security))]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()

    G.print_nodes()

    print("\nName to vertex ID map entries")
    G.print_name_to_vertex()

    G.print_all_paths()

    print("\nRemoving a several concepts, some valid, some invalid")
    G.remove_nodes(concepts=set(["invalid1", tension, "invalid2"]))
    G.print_nodes()
    print("\nName to vertex ID map entries")
    G.print_name_to_vertex()
    G.print_all_paths()


def test_remove_edge():
    causal_fragments = [(("small", 1, tension), ("large", -1, food_security))]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()

    G.print_nodes()
    G.print_all_paths()

    print("\nRemoving edge - invalid source")
    with pytest.raises(IndexError):
        G.remove_edge(source="invalid", target=food_security)

    print("\nRemoving edge - invalid target")
    with pytest.raises(IndexError):
        G.remove_edge(source=tension, target="invalid")

    print("\nRemoving edge - source and target inverted target")
    G.remove_edge(source=food_security, target=tension)
    G.print_nodes()

    print("\nRemoving edge - correct")
    G.remove_edge(source=tension, target=food_security)
    G.print_nodes()
    G.print_edges()
    G.to_png()


def test_remove_edges():
    causal_fragments = [(("small", 1, tension), ("large", -1, food_security))]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()

    G.print_nodes()

    print("\nName to vertex ID map entries")
    G.print_name_to_vertex()

    G.print_all_paths()

    edges_to_remove = [
        ("invalid_src_1", food_security),
        ("invalid_src_2", food_security),
        (tension, "invalid_tgt1"),
        (tension, "invalid_tgt2"),
        ("invalid_src_2", "invalid_tgt_2"),
        ("invalid_src_3", "invalid_tgt3"),
        (food_security, tension),
        (tension, food_security),
    ]
    print("\nRemoving edges")
    G.remove_edges(edges_to_remove)
    G.print_nodes()
    print("\nName to vertex ID map entries")
    G.print_name_to_vertex()
    G.print_all_paths()


def test_subgraph():
    causal_fragments = [  # Center node is n4
        (("small", 1, "n0"), ("large", -1, "n1")),
        (("small", 1, "n1"), ("large", -1, "n2")),
        (("small", 1, "n2"), ("large", -1, "n3")),
        (("small", 1, "n3"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n5")),
        (("small", 1, "n5"), ("large", -1, "n6")),
        (("small", 1, "n6"), ("large", -1, "n7")),
        (("small", 1, "n7"), ("large", -1, "n8")),
        (("small", 1, "n0"), ("large", -1, "n9")),
        (("small", 1, "n9"), ("large", -1, "n2")),
        (("small", 1, "n2"), ("large", -1, "n10")),
        (("small", 1, "n10"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n11")),
        (("small", 1, "n11"), ("large", -1, "n6")),
        (("small", 1, "n6"), ("large", -1, "n12")),
        (("small", 1, "n12"), ("large", -1, "n8")),
        (("small", 1, "n13"), ("large", -1, "n14")),
        (("small", 1, "n14"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n15")),
        (("small", 1, "n15"), ("large", -1, "n16")),
        (("small", 1, "n5"), ("large", -1, "n3")),  # Creates a loop
    ]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()

    G.print_nodes()

    print("\nName to vertex ID map entries")
    G.print_name_to_vertex()

    G.print_nodes()
    G.print_name_to_vertex()

    hops = 2
    node = "n4"
    print(
        "\nSubgraph of {} hops beginning at node {} graph".format(hops, node)
    )
    try:
        G_sub = G.get_subgraph_for_concept(node, False, hops)
    except IndexError:
        print("Concept {} is not in the CAG!".format(node))
        return

    print("\n\nTwo Graphs")
    print("The original")
    G.print_nodes()
    G.print_name_to_vertex()

    print("The subgraph")
    G_sub.print_nodes()
    G_sub.print_name_to_vertex()

    print("\nSubgraph of {} hops ending at node {} graph".format(hops, node))
    G_sub = G.get_subgraph_for_concept(node, True, hops)

    print("\n\nTwo Graphs")
    print("The original")
    G.print_nodes()
    G.print_name_to_vertex()

    print(
        "\nSubgraph of {} hops beginning at node {} graph".format(hops, node)
    )
    G.get_subgraph_for_concept(node, False, hops)


def test_subgraph_between():
    causal_fragments = [  # Center node is n4
        (("small", 1, "n0"), ("large", -1, "n1")),
        (("small", 1, "n1"), ("large", -1, "n2")),
        (("small", 1, "n2"), ("large", -1, "n3")),
        (("small", 1, "n3"), ("large", -1, "n4")),
        (("small", 1, "n0"), ("large", -1, "n5")),
        (("small", 1, "n5"), ("large", -1, "n6")),
        (("small", 1, "n6"), ("large", -1, "n4")),
        (("small", 1, "n0"), ("large", -1, "n7")),
        (("small", 1, "n7"), ("large", -1, "n4")),
        (("small", 1, "n0"), ("large", -1, "n4")),
        (("small", 1, "n0"), ("large", -1, "n8")),
        (("small", 1, "n8"), ("large", -1, "n9")),
        (("small", 1, "n10"), ("large", -1, "n0")),
        (("small", 1, "n4"), ("large", -1, "n12")),
        (("small", 1, "n12"), ("large", -1, "n13")),
        (("small", 1, "n13"), ("large", -1, "n4")),
    ]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()

    G.print_nodes()

    print("\nName to vertex ID map entries")
    G.print_name_to_vertex()

    G.print_nodes()
    G.print_name_to_vertex()
    # G.print_all_paths()

    cutoff = 3
    src = "n0"
    tgt = "n4"

    print(
        "\nSubgraph with inbetween hops less than or equal {} between source node {} and target node {}".format(
            cutoff, src, tgt
        )
    )
    try:
        G_sub = G.get_subgraph_for_concept_pair(src, tgt, cutoff)
        # G_sub.find_all_paths()
    except IndexError:
        print("Incorrect source or target concept")
        return

    print("\n\nTwo Graphs")
    print("The original")
    G.print_nodes()
    G.print_name_to_vertex()
    print()

    print("The subgraph")
    G_sub.print_nodes()
    G_sub.print_name_to_vertex()
    # G_sub.print_all_paths()


def test_prune():
    causal_fragments = [  # Center node is n4
        (("small", 1, "n0"), ("large", -1, "n1")),
        (("small", 1, "n0"), ("large", -1, "n2")),
        (("small", 1, "n0"), ("large", -1, "n3")),
        (("small", 1, "n2"), ("large", -1, "n1")),
        (("small", 1, "n3"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n1")),
        # (("small", 1, "n4"), ("large", -1, "n2")),
        # (("small", 1, "n2"), ("large", -1, "n3")),
    ]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()

    print("\nBefore pruning")
    G.print_all_paths()

    cutoff = 2

    G.prune(cutoff)

    print("\nAfter pruning")
    G.print_all_paths()


def test_merge():
    causal_fragments = [
        (("small", 1, tension), ("large", -1, food_security)),
        (("small", 1, displacement), ("small", 1, tension)),
        (("small", 1, displacement), ("large", -1, food_security)),
        (("small", 1, tension), ("small", 1, crop_production)),
        (("large", -1, food_security), ("small", 1, crop_production)),
        (
            ("small", 1, "UN/events/human/economic_crisis"),
            ("small", 1, tension),
        ),
        (
            ("small", 1, "UN/events/weather/precipitation"),
            ("large", -1, food_security),
        ),
        (("large", -1, food_security), ("small", 1, inflation)),
    ]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()

    print("\nBefore merging")
    G.print_all_paths()

    G.print_nodes()

    print("\nAfter merging")
    G.merge_nodes(food_security, tension)

    G.print_all_paths()

    G.print_nodes()


def test_debug():
    causal_fragments = [  # Center node is n4
        (("small", 1, "n0"), ("large", -1, "n1")),
        (("small", 1, "n0"), ("large", -1, "n2")),
        (("small", 1, "n0"), ("large", -1, "n3")),
        (("small", 1, "n2"), ("large", -1, "n1")),
        (("small", 1, "n3"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n1")),
    ]

    causal_fragments = [  # Center node is n4
        (("small", 1, "n0"), ("large", -1, "n1")),
        (("small", 1, "n1"), ("large", -1, "n2")),
        (("small", 1, "n2"), ("large", -1, "n3")),
        (("small", 1, "n3"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n5")),
        (("small", 1, "n5"), ("large", -1, "n6")),
        (("small", 1, "n6"), ("large", -1, "n7")),
        (("small", 1, "n7"), ("large", -1, "n8")),
        (("small", 1, "n0"), ("large", -1, "n9")),
        (("small", 1, "n9"), ("large", -1, "n2")),
        (("small", 1, "n2"), ("large", -1, "n10")),
        (("small", 1, "n10"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n11")),
        (("small", 1, "n11"), ("large", -1, "n6")),
        (("small", 1, "n6"), ("large", -1, "n12")),
        (("small", 1, "n12"), ("large", -1, "n8")),
        (("small", 1, "n13"), ("large", -1, "n14")),
        (("small", 1, "n14"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n15")),
        (("small", 1, "n15"), ("large", -1, "n16")),
        (("small", 1, "n5"), ("large", -1, "n3")),  # Creates a loop
    ]

    causal_fragments = [  # Center node is n4
        (("small", 1, "n0"), ("large", -1, "n1")),
        (("small", 1, "n1"), ("large", -1, "n2")),
        (("small", 1, "n2"), ("large", -1, "n3")),
        (("small", 1, "n3"), ("large", -1, "n4")),
        (("small", 1, "n4"), ("large", -1, "n5")),
        (("small", 1, "n5"), ("large", -1, "n6")),
        (("small", 1, "n6"), ("large", -1, "n7")),
        (("small", 1, "n7"), ("large", -1, "n8")),
        (("small", 1, "n0"), ("large", -1, "n3")),
    ]

    print("\n\n\n\n")
    print("\nCreating CAG")
    G = AnalysisGraph.from_causal_fragments(causal_fragments)
    G.find_all_paths()
    G.print_nodes()

    print("\nBefore pruning")
    G.print_all_paths()

    hops = 3
    node = "n0"
    print(
        f"\nSubgraph of {hops} hops beginning at node {node} graph"
    )
    try:
        G_sub = G.get_subgraph_for_concept(node, False, hops)
    except IndexError:
        print(f"Concept {node} is not in the CAG!")
        return

    G_sub.find_all_paths()
    G_sub.print_nodes()
