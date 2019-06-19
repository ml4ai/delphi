from AnalysisGraph import AnalysisGraph
G = AnalysisGraph.from_json_file("indra_statements_format.json")
G.construct_beta_pdfs()
