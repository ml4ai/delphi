from delphi import AnalysisGraph

G = AnalysisGraph.from_text(
    "Significantly increased conflict seen in South Sudan forced many families"
    "to flee in 2017.")
G.map_concepts_to_indicators()
G.parameterize(country="South Sudan", year=2017, month=4)
A = G.to_agraph()
A.draw("CAG.png", prog="dot")
