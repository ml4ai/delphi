from delphi import AnalysisGraph

G = AnalysisGraph.from_text("Conflict increases displacement.")
A = G.to_agraph()
A.draw("CAG.png", prog="dot")
