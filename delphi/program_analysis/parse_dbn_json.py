from os.path import normpath
from pygraphviz import AGraph
from typing import Dict
import json

import delphi.program_analysis.scopes as scp


def main():
    dbn_json_source_file = normpath("autoTranslate/pgm.json")
    data = read_dbn_from_json(dbn_json_source_file)

    scope_tree = scp.scope_tree_from_json(data)

    g2 = setup_directed_graph()
    scope_tree.build_containment_graph(g2)
    g2.write("nested_graph.dot")
    g2.draw("nested_graph.png", prog="dot")


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
