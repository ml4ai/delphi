# PURPOSE: Creates a directed dependency graph to visualize the module
#          dependencies extracted from a Fortran codebase
# CREATOR: Paul Hein
# USAGE: python dependency_viewer.py /path/to/coverage_file_map.json

import json
import sys

import networkx as nx

# NOTE: Pass in the path to the coverage JSON file as an input argument
module_data = json.load(open(sys.argv[1], "r"))
G = nx.DiGraph()
G.add_edges_from([
    (key, mod_name) for key, data in module_data.items()
    for mod_name in data["Imported modules"]
])
A = nx.nx_agraph.to_agraph(G)

# NOTE: Pygraphviz and graphviz are needed to do the drawing in this step
A.draw("DSSAT_dependency_graph.pdf", prog="circo")
