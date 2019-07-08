from delphi.cpp.AnalysisGraph import AnalysisGraph

def test_cpp_extensions():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    G.construct_beta_pdfs()
    G.sample_from_prior()

def test_simple_path_construction():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    G.add_node()
    G.add_node()
    G.add_node()
    G.add_node()

    print( 'Nodes of the graph:' )
    G.print_nodes()

    G.add_edge(0,1)
    G.add_edge(1,2)
    #G.add_edge(1,3)
    #G.add_edge(2,3)
    G.add_edge(0,2)
    G.add_edge(3,1) # Creates a loop 1 -> 2 -> 3 -> 1

    print( 'Edges of the graph:' )
    G.print_edges()

    G.find_all_paths()
    G.print_all_paths()

    G.print_cells_affected_by_beta( 0, 1 )
    G.print_cells_affected_by_beta( 1, 2 )
