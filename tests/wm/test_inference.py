import pytest
from pytest import approx
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
from conftest import G, concepts
import seaborn as sns

conflict_string = concepts['conflict']['grounding']
food_security_string = concepts['food security']['grounding']
human_migration_string = concepts['migration']['grounding']
product_string = concepts['product']['grounding']

@pytest.mark.skip
def test_inference_with_synthetic_data(G):
    """ Smokescreen test for sampler. """
    # Generate synthetic data

    # Sample a transition matrix from the prior
    A = G.sample_from_prior()[0]

    # Get the original value of our parameter of interest (the ground truth
    # value that we can use to evaluate our inference.
    original_beta = A[f"∂({conflict_string})/∂t"][food_security_string]
    # original_beta_2 = A[f"∂({human_migration_string})/∂t"][product_string]
    fig, ax = plt.subplots()

    # Initialize the latent state vector at time 0
    G.s0 = G.construct_default_initial_state()
    for edge in G.edges(data=True):
        G.s0[f"∂({edge[0]})/∂t"] = 0.1*np.random.rand()

    # Given the initial latent state vector and the sampled transition matrix,
    # sample a sequence of latent states and observed states
    G.sample_from_likelihood(n_timesteps=10)
    G.latent_state_sequence = G.latent_state_sequences[0]
    G.observed_state_sequence = G.observed_state_sequences[0]

    # Create empty lists to hold the scores
    betas = []
    # betas_2 = []
    log_likelihoods = []
    log_priors = []
    map_estimates=[]


    for edge in G.edges(data=True):
        A[f"∂({edge[0]})/∂t"][edge[1]] = 0.0

    n_samples: int = 1000
    for i, _ in enumerate(trange(n_samples)):
        G.sample_from_posterior(A)
        betas.append(A[f"∂({conflict_string})/∂t"][food_security_string])
        # betas_2.append(A[f"∂({human_migration_string})/∂t"][product_string])

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

@pytest.mark.skip
def test_inference_with_real_data(G):
    G.get_timeseries_values_for_indicators()
