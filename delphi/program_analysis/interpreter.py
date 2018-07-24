import platform
from os.path import normpath
from pygraphviz import AGraph
from typing import Dict
import json

import delphi.program_analysis.scopes as scp

def main():
    dbn_json = json.load(open(normpath("autoTranslate/pgm.json")))

    scope_types_dict = {'container': scp.FuncScope, 'loop_plate': scp.LoopScope}

    scopes = {f['name']: scope_types_dict[f['type']](f['name'], f)
              for f in dbn_json['functions']
              if f['type'] in scope_types_dict}

    # Make a list of all scopes by scope names
    scope_names = list(scopes.keys())

    # Remove pseudo-scopes we wish to not display (such as print)
    for scope in scopes.values():
        scope.remove_non_scope_children(scope_names)

    # Build the nested tree of scopes using recursion
    root = scopes[dbn_json["start"]]
    root.build_scope_tree(scopes)
    root.setup_from_json()

    A = AGraph(directed=True)
    A.node_attr["shape"] = "rectangle"
    A.graph_attr["rankdir"] = "LR"

    operating_system = platform.system()

    if operating_system == 'Darwin':
        font = "Menlo"
    elif operating_system == 'Windows':
        font = "Consolas"
    else:
        font = "Courier"

    A.node_attr["fontname"] = font
    A.graph_attr["fontname"] = font

    root.build_containment_graph(A)
    A.write("nested_graph.dot")
    A.draw("nested_graph.png", prog="dot")


if __name__ == '__main__':
    main()
