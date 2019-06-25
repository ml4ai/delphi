from delphi.cpp.AnalysisGraph import AnalysisGraph
G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
G.construct_beta_pdfs()
