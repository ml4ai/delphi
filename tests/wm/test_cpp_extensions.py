import numpy as np
import pytest
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from delphi.cpp.DelphiPython import AnalysisGraph, Indicator


def test_cpp_extensions():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    #G.construct_beta_pdfs()

    #G.print_nodes()

    #print( '\nName to vertex ID map entries' )
    #G.print_name_to_vertex()

    #G.sample_from_prior()

def test_simple_path_construction():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    G.add_node('c0')
    G.add_node('c1')
    G.add_node('c2')
    #G.add_node('c3')
    #G.add_node('c4')

    print( 'Nodes of the graph:' )
    G.print_nodes()

    G.add_edge((("", 1, "c0"), ("", 1, "c1")))
    G.add_edge((("", 1, "c1"), ("", 1, "c2")))
    #G.add_edge((("", 1, "c1"), ("", 1, "c3")))
    #G.add_edge((("", 1, "c2"), ("", 1, "c3")))
    G.add_edge((("", 1, "c0"), ("", 1, "c2")))
    G.add_edge((("", 1, "c3"), ("", 1, "c1")))# Creates a loop 1 -> 2 -> 3 -> 1
    '''
    G.add_edge(0,1)
    G.add_edge(1,2)
    #G.add_edge(1,3)
    #G.add_edge(2,3)
    G.add_edge(0,2)
    G.add_edge(3,1) # Creates a loop 1 -> 2 -> 3 -> 1
    '''

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
    G.remove_nodes( concepts = set(['invalid1', 'UN/events/human/conflict', 'invalid2' ]) )
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
    G.to_png()

def test_remove_edges():
    causal_fragments = [ (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security"))]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    G.print_all_paths()

    edges_to_remove = [( 'invalid_src_1', "UN/entities/human/food/food_security"),
                       ( 'invalid_src_2', "UN/entities/human/food/food_security"),
                       ( 'UN/events/human/conflict', 'invalid_tgt1'),
                       ( 'UN/events/human/conflict', 'invalid_tgt2'),
                       ( 'invalid_src_2', 'invalid_tgt_2'),
                       ( 'invalid_src_3', 'invalid_tgt3'),
                       ( "UN/entities/human/food/food_security", 'UN/events/human/conflict'),
                       ( 'UN/events/human/conflict', "UN/entities/human/food/food_security"),
                      ]
    print( '\nRemoving edges' )
    G.remove_edges(edges_to_remove)
    G.print_nodes()
    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()
    G.print_all_paths()

def test_subgraph():
    causal_fragments = [  # Center node is n4
            (("small", 1, "n0"), ("large", -1, "n1")),
            (("small", 1, "n1"), ("large", -1, "n2")),
            (("small", 1, "n2"), ("large", -1, "n3")),
            (("small", 1, "n3"), ("large", -1, "n4")),
            (("small", 1, "n4"), ("large", -1, "n5")),
            (("small", 1, "n5"), ("large", -1, "n6")),
            (("small", 1, "n6"), ("large", -1, "n7")),
            (("small", 1, "n7"), ("large", -1, "n8")),
            #(("small", 1, "n8"), ("large", -1, "n9")),
            #(("small", 1, "n9"), ("large", -1, "n0")),
            (("small", 1, "n0"), ("large", -1, "n9")),
            (("small", 1, "n9"), ("large", -1, "n2")),
            (("small", 1, "n2"), ("large", -1, "n10")),
            (("small", 1, "n10"), ("large", -1, "n4")),
            (("small", 1, "n4"), ("large", -1, "n11")),
            (("small", 1, "n11"), ("large", -1, "n6")),
            (("small", 1, "n6"), ("large", -1, "n12")),
            (("small", 1, "n12"), ("large", -1, "n8")),
            (("small", 1, "n13"), ("large", -1, "n14")),
            (("small", 1, "n14"), ("large", -1, "n4")),
            (("small", 1, "n4"), ("large", -1, "n15")),
            (("small", 1, "n15"), ("large", -1, "n16")),
            (("small", 1, "n5"), ("large", -1, "n3")), # Creates a loop
            ]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    #G.remove_nodes(set(['n0', 'n1', 'n2', 'n3', 'n4']))
    #G.remove_nodes(set(['n2', 'n3', 'n4']))
    #G.remove_nodes(set(['n9', 'n8', 'n7', 'n6', 'n5']))

    G.print_nodes()
    G.print_name_to_vertex()

    hops = 3
    node = 'n40'
    print( '\nSubgraph of {} hops beginning at node {} graph'.format( hops, node ) )
    try:
        G_sub = G.get_subgraph_for_concept( node, hops, False )
    except IndexError:
        print('Concept {} is not in the CAG!'.format(node))
        return

    print( '\n\nTwo Graphs' )
    print( 'The original' )
    G.print_nodes()
    G.print_name_to_vertex()
    #G.print_all_paths()
    print()


    print( 'The subgraph' )
    G_sub.print_nodes()
    G_sub.print_name_to_vertex()
    #G_sub.print_all_paths()

    print( '\nSubgraph of {} hops ending at node {} graph'.format( hops, node ) )
    G_sub = G.get_subgraph_for_concept( node, hops, True )

    print( '\n\nTwo Graphs' )
    print( 'The original' )
    G.print_nodes()
    G.print_name_to_vertex()
    #G.print_all_paths()
    print()


    print( 'The subgraph' )
    G_sub.print_nodes()
    G_sub.print_name_to_vertex()
    #G_sub.print_all_paths()

def test_subgraph_between():
    causal_fragments = [  # Center node is n4
            (("small", 1, "n0"), ("large", -1, "n1")),
            (("small", 1, "n1"), ("large", -1, "n2")),
            (("small", 1, "n2"), ("large", -1, "n3")),
            (("small", 1, "n3"), ("large", -1, "n4")),
            (("small", 1, "n4"), ("large", -1, "n5")),
            (("small", 1, "n5"), ("large", -1, "n6")),
            (("small", 1, "n6"), ("large", -1, "n7")),
            (("small", 1, "n7"), ("large", -1, "n8")),
            #(("small", 1, "n8"), ("large", -1, "n9")),
            #(("small", 1, "n9"), ("large", -1, "n0")),
            (("small", 1, "n0"), ("large", -1, "n9")),
            (("small", 1, "n9"), ("large", -1, "n2")),
            (("small", 1, "n2"), ("large", -1, "n10")),
            (("small", 1, "n10"), ("large", -1, "n4")),
            (("small", 1, "n4"), ("large", -1, "n11")),
            (("small", 1, "n11"), ("large", -1, "n6")),
            (("small", 1, "n6"), ("large", -1, "n12")),
            (("small", 1, "n12"), ("large", -1, "n8")),
            (("small", 1, "n13"), ("large", -1, "n14")),
            (("small", 1, "n14"), ("large", -1, "n4")),
            (("small", 1, "n4"), ("large", -1, "n15")),
            (("small", 1, "n15"), ("large", -1, "n16")),
            (("small", 1, "n5"), ("large", -1, "n3")), # Creates a loop
            ]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    G.print_nodes()
    G.print_name_to_vertex()

    cutoff = 12
    src = 'n0'
    tgt = 'n8'

    print( '\nSubgraph with inbetween hops less than or equal {} between source node {} and target node {}'.format( cutoff, src, tgt ) )
    try:
        G_sub = G.get_subgraph_for_concept_pair( src, tgt, cutoff )
    except IndexError:
        print("Incorrect source or target concept")
        return

    print( '\n\nTwo Graphs' )
    print( 'The original' )
    G.print_nodes()
    G.print_name_to_vertex()
    #G.print_all_paths()
    print()


    print( 'The subgraph' )
    G_sub.print_nodes()
    G_sub.print_name_to_vertex()
    #G_sub.print_all_paths()

def test_prune():
    causal_fragments = [  # Center node is n4
            (("small", 1, "n0"), ("large", -1, "n1")),
            (("small", 1, "n0"), ("large", -1, "n2")),
            (("small", 1, "n0"), ("large", -1, "n3")),
            (("small", 1, "n2"), ("large", -1, "n1")),
            (("small", 1, "n3"), ("large", -1, "n4")),
            (("small", 1, "n4"), ("large", -1, "n1")),
            #(("small", 1, "n4"), ("large", -1, "n2")),
            #(("small", 1, "n2"), ("large", -1, "n3")),
            ]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    print( '\nBefore pruning' )
    G.print_all_paths()

    cutoff = 2

    G.prune( cutoff )

    print( '\nAfter pruning' )
    G.print_all_paths()

def test_merge():
    causal_fragments = [ 
            (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security")),
            (("small", 1, "UN/events/human/human_migration"), ("small", 1, "UN/events/human/conflict")),
            (("small", 1, "UN/events/human/human_migration"), ("large", -1, "UN/entities/human/food/food_security")),
            (("small", 1, "UN/events/human/conflict") , ("small", 1, "UN/entities/natural/crop_technology/product")),
            (("large", -1, "UN/entities/human/food/food_security") , ("small", 1, "UN/entities/natural/crop_technology/product")),
            (("small", 1, "UN/events/human/economic_crisis"), ("small", 1, "UN/events/human/conflict")),
            (("small", 1, "UN/events/weather/precipitation"), ("large", -1, "UN/entities/human/food/food_security")),
            (("small", 1, "UN/entities/human/financial/economic/inflation"), ("small", 1, "UN/events/human/conflict")),
            ( ("large", -1, "UN/entities/human/food/food_security"), ("small", 1, "UN/entities/human/financial/economic/inflation")),
            ]

    '''
    ("large", -1, "UN/entities/human/food/food_security")
    ("small", 1, "UN/events/human/conflict")
    ("small", 1, "UN/events/human/human_migration")
    ("small", 1, "UN/entities/natural/crop_technology/product")
    ("small", 1, "UN/events/human/economic_crisis")
    ("small", 1, "UN/events/weather/precipitation")
    ("small", 1, "UN/entities/human/financial/economic/inflation")
    '''
    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_causal_fragments( causal_fragments )

    print( '\nBefore merging' )
    G.print_all_paths()

    G.print_nodes()

    print('\nAfter mergning')
    #G.merge_nodes( "UN/events/human/conflict", "UN/entities/human/food/food_security")
    G.merge_nodes( "UN/entities/human/food/food_security", "UN/events/human/conflict")

    G.print_all_paths()

    G.print_nodes()
