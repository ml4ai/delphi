from delphi.cpp.AnalysisGraph import AnalysisGraph

def test_cpp_extensions():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    G.construct_beta_pdfs()
    G.sample_from_prior()
