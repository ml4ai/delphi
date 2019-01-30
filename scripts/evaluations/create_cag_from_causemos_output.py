import sys
import pickle
from delphi.AnalysisGraph import AnalysisGraph
from delphi.export import to_agraph

G = AnalysisGraph.from_uncharted_json_file(sys.argv[1])
A = to_agraph(G, indicators=True, indicator_values = True)
A.draw("CauseMos_CAG.pdf", prog="dot")

with open(sys.argv[2], 'wb') as f:
    pickle.dump(G, f)
