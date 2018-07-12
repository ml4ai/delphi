from os.path import join, normpath
from pygraphviz import AGraph
from typing import Dict
import pprint
import json

import scopes as scp


def main():
    example_source_file = "crop_yield_DBN.json"
    filepath = "../../data/program_analysis/pa_crop_yield_v0.2/"
    dbn_json_source_file = normpath(join(filepath, example_source_file))
    data = read_dbn_from_json(dbn_json_source_file)

    # print('DBN_JSON:')
    # pprint.pprint(data)

    scopes = build_scopes(data)
    scope_names = list(scopes.keys())

    for scope in scopes.values():
        scope.remove_non_scope_children(scope_names)

    root = find_outermost_scope(scopes, scope_names)
    root.build_scope_tree(scopes)

    A = AGraph(directed=True)
    A.node_attr["shape"] = "rectangle"
    A.graph_attr["rankdir"] = "LR"
    A.node_attr["fontname"] = "Gill Sans"

    root.build_linked_graph(A)

    A.draw("linked_graph.png", prog="dot")

    B = AGraph(directed=True)
    B.node_attr["shape"] = "rectangle"
    B.graph_attr["rankdir"] = "LR"
    B.node_attr["fontname"] = "Gill Sans"

    root.build_containment_graph(B)

    B.draw("nested_graph.png", prog="dot")


def read_dbn_from_json(dbn_json_source_file: str) -> Dict:
    """
    Read DBN from JSON file.
    :param dbn_json_source_file: JSON source filename in DBN.JSON format
    :return: Dict representing JSON
    """

    with open(dbn_json_source_file) as fin:
        dbn_json = json.load(fin)

    return dbn_json


def build_scopes(json_data: Dict) -> Dict:
    """
    Using input data from JSON find all function and loop_plate objects. Build
    a new scope object for each of these appropriately. Index the new scope
    into a dictionary of scopes using the name of the scope as the key.

    :param json_data: input data from JSON that contains scope definitions
    :return: A dictionary of all discovered scopes
    """
    result = dict()
    for f in json_data["functions"]:
        if f["type"] == "container":
            result[f["name"]] = scp.FuncScopeNode(f["name"], f)
        elif f["type"] == "loop_plate":
            result[f["name"]] = scp.LoopScopeNode(f["name"], f)

    return result


def find_outermost_scope(scopes: Dict, scope_names: list) -> scp.ScopeNode:
    for name in scope_names:
        dependent = False
        for c1 in scopes.values():
            if name in c1.child_names:
                dependent = True
                break
        if not dependent:
            return scopes[name]


if __name__ == "__main__":
    main()
