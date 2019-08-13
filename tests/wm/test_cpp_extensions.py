import numpy as np
import pytest
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from delphi.cpp.AnalysisGraph import AnalysisGraph, Indicator


def test_cpp_extensions():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    #G.construct_beta_pdfs()

    #G.print_nodes()

    #print( '\nName to vertex ID map entries' )
    #G.print_name_to_vertex()

    #G.sample_from_prior()

def test_simple_path_construction():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    G.add_node()
    G.add_node()
    #G.add_node()
    #G.add_node()

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

    G2 = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")

    #G2.initialize( True )
    '''
    samples = G2.sample_from_prior()

    print( 'Nunber of samples from prior: ', len(samples) )
    for i in range(5):
        print( samples[i] )
    '''
    #G2.sample_from_likelihood( 10 )

    #G.sample_from_proposal_debug()


def test_inference():
    causal_fragments = [ (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security"))]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    print( '\nSample from proposal debug' )
    #G.sample_from_proposal_debug()

def test_remove_node():
    causal_fragments = [ (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security"))]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    G.print_all_paths()

    print( '\nRemoving an invalid concept' )
    G.remove_node( concept = 'invalid' )
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()

    print( '\nRemoving a valid concept' )
    G.remove_node( concept = 'UN/events/human/conflict' )
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()

def test_remove_nodes():
    causal_fragments = [ (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security"))]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    G.print_all_paths()

    print( '\nRemoving a several concepts, some valid, some invalid' )
    G.remove_nodes( concepts = ['invalid1', 'UN/events/human/conflict', 'invalid2' ] )
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()

def test_remove_edge():
    causal_fragments = [ (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security"))]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    G.print_all_paths()

    print( '\nRemoving edge - invalid source' )
    G.remove_edge( source = 'invalid', target = "UN/entities/human/food/food_security" )
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()

    print( '\nRemoving edge - invalid target' )
    G.remove_edge( source = 'UN/events/human/conflict', target = 'invalid' )
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()

    print( '\nRemoving edge - source and target inverted target' )
    G.remove_edge( source = "UN/entities/human/food/food_security", target = 'UN/events/human/conflict' )
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()


    print( '\nRemoving edge - correct' )
    G.remove_edge( source = 'UN/events/human/conflict', target = "UN/entities/human/food/food_security")
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()
