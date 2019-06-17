import pytest
from pytest import approx
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
from conftest import G, concepts
import seaborn as sns

@pytest.mark.skip
def test_inference_with_synthetic_data(G):
    """ Smokescreen test for sampler. """
    # Generate synthetic data

    # Sample a transition matrix from the prior
    A = G.transition_matrix_collection[0]

    # Get the original value of our parameter of interest (the ground truth
    # value that we can use to evaluate our inference.
    original_beta = A[f"∂({conflict_string})/∂t"][f"{food_security_string}"]
    fig, ax = plt.subplots()
    sns.distplot(G.edges[conflict_string,
        food_security_string]["βs"], ax=ax)
    plt.savefig("betas_dist.pdf")

    # Initialize the latent state vector at time 0
    G.s0 = G.construct_default_initial_state()
    G.s0[f"∂({conflict_string})/∂t"] = 0.1

    # Given the initial latent state vector and the sampled transition matrix,
    # sample a sequence of latent states and observed states
    G.sample_from_likelihood()
    G.latent_state_sequence = G.latent_state_sequences[0]
    G.observed_state_sequence = G.observed_state_sequences[0]

    # Perform an initial calculation of the log prior and the log likelihood.
    G.update_log_prior(A)
    G.update_log_likelihood()
    G.update_log_joint_probability()
    original_score = G.log_joint_probability

    # Initialize random walk
    A[f"∂({conflict_string})/∂t"][food_security_string] = 0.1

    # Save the current MAP estimate of the beta parameter (i.e. the value that
    # we are starting with.
    map_estimate = A[f"∂({conflict_string})/∂t"][food_security_string]
    map_log_joint_probability = original_score


    # Create empty lists to hold the scores
    scores = []
    betas = []
    log_likelihoods = []
    log_priors = []
    map_estimates = []


    n_samples: int = 10000
    for i, _ in enumerate(trange(n_samples)):
        G.sample_from_posterior(A)
        if G.log_joint_probability > map_log_joint_probability:
            map_estimate = A[f"∂({conflict_string})/∂t"][food_security_string]
            map_log_joint_probability = G.log_joint_probability
        scores.append(G.log_joint_probability - original_score)
        betas.append(A[f"∂({conflict_string})/∂t"][food_security_string])
        log_likelihoods.append(G.log_likelihood)
        log_priors.append(G.log_prior)
        map_estimates.append(map_estimate)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(betas[::10], label="beta")
    axes[0].plot(
        original_beta * np.ones(len(betas[::10])), label="original_beta"
    )
    axes[1].plot(log_priors[::10], label="log_prior")
    axes[2].plot(log_likelihoods[::10], label="log_likelihood")
    for ax in axes:
        ax.legend()
    plt.savefig("mcmc_results.pdf")
    fig, ax = plt.subplots()
    plt.hist(betas, bins=40)
    plt.savefig("betas.pdf")

    # This tolerance seems to work for now, so I'm leaving it in.
    assert map_estimate == approx(original_beta, abs=0.1)

def test_inference_with_real_data(G):
    G.get_timeseries_values_for_indicators()
