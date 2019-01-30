import pytest
from pytest import approx
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
from conftest import G, conflict_string, food_security_string

def test_inference(G):

    # Generate synthetic data
    A = G.transition_matrix_collection[0]
    original_beta = A[f"∂({conflict_string})/∂t"][f"{food_security_string}"]
    G.s0 = G.construct_default_initial_state()
    G.s0[f"∂({conflict_string})/∂t"] = 0.1
    G.sample_from_likelihood()
    G.latent_state_sequence = G.latent_state_sequences[0]
    G.observed_state_sequence = G.observed_state_sequences[0]
    G.update_log_prior(A)
    G.update_log_likelihood()
    G.update_log_joint_probability()
    original_score = G.log_joint_probability

    # Initialize random walk
    A[f"∂({conflict_string})/∂t"][food_security_string] = -0.01
    map_estimate = A[f"∂({conflict_string})/∂t"][food_security_string]
    map_log_joint_probability =original_score


    scores = []
    betas = []
    log_likelihoods = []
    log_priors = []
    map_estimates = []


    n_samples: int = 1000
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
    assert map_estimate == approx(original_beta, abs=0.2)
    assert True

@pytest.mark.skip
def test_sampler(G):
    """ Smokescreen test for sampler. """

    n_samples = 1000


    # Generate synthetic data
    G.sample_from_prior()
    original_beta = -0.2
    sampler.A[f"∂({conflict_string})/∂t"][food_security_string] = original_beta
    sampler.s0[f"∂({conflict_string})/∂t"] = 0.1
    sampler.set_latent_state_sequence()
    sampler.sample_from_likelihood()


    # Initialize random walk



