from os.path import join, normpath
from pygraphviz import AGraph
from typing import Dict
import pprint
import json

import delphi.program_analysis.scopes as scp


def main():
    example_source_file = "crop_yield_DBN.json"
    filepath = "../../data/program_analysis/pa_crop_yield_v0.2/"
    dbn_json_source_file = normpath(join(filepath, example_source_file))
    data = read_dbn_from_json(dbn_json_source_file)

    # print('DBN_JSON:')
    # pprint.pprint(data)

    scope_tree = scp.scope_tree_from_json(data)

    g1 = setup_directed_graph()
    scope_tree.build_linked_graph(g1)
    g1.write("linked_graph.dot")

    g2 = setup_directed_graph()
    scope_tree.build_containment_graph(g2)
    g2.write("nested_graph.dot")


def read_dbn_from_json(dbn_json_source_file: str) -> Dict:
    """
    Read DBN from JSON file.
    :param dbn_json_source_file: JSON source filename in DBN.JSON format
    :return: Dict representing JSON
    """

    with open(dbn_json_source_file) as fin:
        dbn_json = json.load(fin)

    return dbn_json


def setup_directed_graph():
    """
    Creates a Graph instance with our desired configuration

    :return: The Graph object
    """
    A = AGraph(directed=True)
    A.node_attr["shape"] = "rectangle"
    A.graph_attr["rankdir"] = "LR"
    A.node_attr["fontname"] = "Menlo"
    A.graph_attr["fontname"] = "Menlo"
    # A.graph_attr["size"] = "12,14"
    return A


if __name__ == "__main__":
    main()
