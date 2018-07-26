from os.path import normpath
from pygraphviz import AGraph
from typing import Dict
import json

import delphi.program_analysis.scopes as scp

if __name__ == '__main__':
    dbn_json_file = "autoTranslate/pgm.json"
    scope = scp.Scope.from_json(normpath(dbn_json_file))
    A = scope.to_agraph()
    A.write("nested_graph.dot")
    A.draw("nested_graph.png", prog="dot")
