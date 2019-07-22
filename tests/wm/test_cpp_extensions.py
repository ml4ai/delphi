from delphi.cpp.AnalysisGraph import AnalysisGraph
import pytest
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import seaborn as sns

def test_cpp_extensions():
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    #G.construct_beta_pdfs()
    G.sample_from_prior()

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

    G2.initialize( True )
    samples = G2.sample_from_prior()

    print( 'Nunber of samples from prior: ', len(samples) )
    for i in range(5):
        print( samples[i] )

    G2.sample_from_likelihood( 10 )

    G.sample_from_proposal_debug()


def test_inference():
    statements = [ (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security"))]

    print('\n\n\n\n')
    print( '\nCreating CAG' )
    G = AnalysisGraph.from_statements( statements )
    #G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")

    G.print_nodes()

    print( '\nName to vertex ID map entries' )
    G.print_name_to_vertex()

    #print( '\nConstructing beta pdfs' )
    #G.construct_beta_pdfs()

    #G.find_all_paths()

    print( '\nSetting initial state' )
    G.set_initial_state()

    print( '\nSample from proposal debug' )
    G.sample_from_proposal_debug()

    # Get the original value of our parameter of interest (the ground truth
    # value that we can use to evaluate our inference.
    original_beta = G.get_beta( "UN/events/human/conflict", "UN/entities/human/food/food_security" )

    print( "\noriginal_beta: ", original_beta )

    print( '\nTaking a step' )
    G.take_step()

    print( '\nTaking second step' )
    G.take_step()

    beta_after_step = G.get_beta( "UN/events/human/conflict", "UN/entities/human/food/food_security" )

    print( "\nbeta after step: ", beta_after_step )

'''
    statements = [ (("small", 1, "UN/events/human/conflict"), ("large", -1, "UN/entities/human/food/food_security"))]

    #G = AnalysisGraph.from_statements( statements )
    G = AnalysisGraph.from_json_file("tests/data/indra_statements_format.json")
    G.print_nodes()
    #G.construct_beta_pdfs()
    #G.find_all_paths()
    G.print_all_paths()
    G.initialize( True )
    samples = G.sample_from_prior()
    G.sample_from_likelihood( 10 )
    G.sample_from_proposal_debug()

    G.set_initial_state()

    # Get the original value of our parameter of interest (the ground truth
    # value that we can use to evaluate our inference.
    original_beta = G.get_beta( "conflict", "food_security" )

    print( "original_beta: ", original_beta )

    G.take_step()
    beta_after_step = G.get_beta( "conflict", "food_security" )

    print( "beta after step: ", beta_after_step )

    # Create empty lists to hold the scores
    betas = []
    log_likelihoods = []
    log_priors = []
    map_estimates=[]

    n_samples: int = 1000
    for i, _ in enumerate(trange(n_samples)):
        #G.sample_from_posterior(A)
        G.take_step()
        betas.append(G.get_beta( "conflict", "food_security" ))

    print( betas )

    fig, ax = plt.subplots()

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(betas, label="beta")
    axes[0].plot(
        original_beta * np.ones(len(betas)), label="original_beta"
    )
    # axes[1].plot(betas_2, label="beta_2")
    # axes[1].plot(
        # original_beta_2 * np.ones(len(betas)), label="original_beta_2"
    # )

    for ax in axes:
        ax.legend()
    plt.savefig("mcmc_results.pdf")
    fig, ax = plt.subplots(1,2,sharey=True, figsize=(6,3))
    sns.distplot(betas, ax=ax[0], norm_hist=True)

    ax[0].set_title(f"$\\beta_{{c, fs}}={original_beta:.3f}$", fontsize=10)
    # sns.distplot(betas_2, ax=ax[1], norm_hist=True)
    # ax[1].set_title(f"$\\beta_{{hm, p}}={original_beta_2:.3f}$", fontsize=10)
    plt.savefig("betas_combined.pdf")

    # This tolerance seems to work for now, so I'm leaving it in.
    # assert map_estimate == approx(original_beta, abs=0.1)
    '''
