from conftest import *
from tqdm import trange
from pytest import approx
from scipy.stats import norm
from delphi.inference import *
import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt


def test_sampler(G):
    """ Smokescreen test for sampler. """

    sampler = Sampler(G)
    sampler.n_timesteps = 10

    n_samples = 1000

    # Generate synthetic data
    sampler.sample_from_prior()
    original_beta = -0.2
    sampler.A[f"∂({conflict_string})/∂t"][food_security_string] = original_beta
    sampler.s0[f"∂({conflict_string})/∂t"] = 0.1
    sampler.set_latent_state_sequence()
    sampler.sample_from_likelihood()

    # Initialize random walk
    sampler.A[f"∂({conflict_string})/∂t"][food_security_string] = 2.0
    sampler.update_log_prior()
    sampler.set_latent_state_sequence()
    sampler.update_log_likelihood()
    sampler.update_log_joint_probability()
    sampler.original_score = sampler.log_joint_probability
    map_estimate = sampler.A[f"∂({conflict_string})/∂t"][food_security_string]
    map_log_joint_probability = sampler.original_score

    scores = []
    betas = []
    log_likelihoods = []
    log_priors = []
    map_estimates = []

    for i, _ in enumerate(trange(n_samples)):
        sampler.sample_from_posterior()
        if sampler.log_joint_probability > map_log_joint_probability:
            map_estimate = sampler.A[f"∂({conflict_string})/∂t"][food_security_string]
        scores.append(sampler.log_joint_probability - sampler.original_score)
        betas.append(sampler.A[f"∂({conflict_string})/∂t"][food_security_string])
        log_likelihoods.append(sampler.log_likelihood)
        log_priors.append(sampler.log_prior)
        map_estimates.append(map_estimate)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(betas[::10], label="beta")
    axes[0].plot(
        original_beta * np.ones(len(betas[::10])), label="original_beta"
    )
    axes[1].plot(map_estimates[::10], label="scores")
    axes[2].plot(log_priors[::10], label="log_prior")
    axes[3].plot(log_likelihoods[::10], label="log_likelihood")
    for ax in axes:
        ax.legend()
    plt.savefig("mcmc_results.pdf")
    fig, ax = plt.subplots()
    plt.hist(betas, bins=40)
    plt.savefig("betas.pdf")
    assert map_estimate == approx(original_beta, abs=0.2)
