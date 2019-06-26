from delphi.cpp.AnalysisGraph import AnalysisGraph

def test_cpp_extensions():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    G.construct_beta_pdfs()
    G.sample_from_prior()
    G.add_node()
    G.add_node()
    G.add_edge(0,1)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(1,3)
    G.print_nodes()
    G.print_edges()
    print(G.simple_paths(0,3))
