from conftest import *
from tqdm import trange
from pytest import approx
from scipy.stats import norm
from delphi.inference import *
import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt


@pytest.mark.slow
@pytest.mark.skip(reason = "Still ironing out some warts in the sampler.")
def test_sampler(G):
    """ Smokescreen test for sampler. """
    sampler = Sampler(G)
    n_timesteps = 2
    n_samples = 100_000
    sampler.s0["∂(conflict)/∂t"] = 0.1
    sampler.sample_from_prior()
    sampler.A["∂(conflict)/∂t"]["food_security"] = -0.1
    sampler.calculate_log_prior()
    original_beta = sampler.A["∂(conflict)/∂t"]["food_security"]
    sampler.A["∂(conflict)/∂t"]["food_security"] = np.random.normal()
    sampler.set_number_of_timesteps(n_timesteps)
    sampler.set_latent_state_sequence()
    sampler.sample_from_likelihood()
    sampler.calculate_log_likelihood()
    sampler.calculate_log_joint_probability()
    sampler.original_score = sampler.log_joint_probability
    map_estimate = original_beta
    map_log_joint_probability = sampler.log_joint_probability
    scores = []
    betas = []
    log_likelihoods = []
    log_priors = []
    for i, _ in enumerate(trange(n_samples)):
        sampler.sample_from_posterior()
        if sampler.log_joint_probability > map_log_joint_probability:
            map_estimate = sampler.A["∂(conflict)/∂t"]["food_security"]
        if i % 100 == 0:
            scores.append(
                sampler.log_joint_probability - sampler.original_score
            )
            betas.append(sampler.A["∂(conflict)/∂t"]["food_security"])
            log_likelihoods.append(sampler.log_likelihood)
            log_priors.append(sampler.log_prior)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(betas, label="beta")
    axes[0].plot(original_beta * np.ones(len(betas)), label="original_beta")
    axes[1].plot(scores, label="scores")
    axes[2].plot(log_priors, label="log_prior")
    axes[3].plot(log_likelihoods, label="log_likelihood")
    for ax in axes:
        ax.legend()
    plt.savefig("mcmc_results.pdf")
    fig, ax = plt.subplots()
    plt.hist(betas)
    plt.savefig("betas.pdf")
    assert map_estimate == approx(original_beta, abs=0.05)
