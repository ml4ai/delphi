from conftest import *
from tqdm import trange
from pytest import approx
from scipy.stats import norm
from delphi.inference import *


@pytest.mark.slow
def test_sampler(G):
    """ Smokescreen test for sampler. """
    sampler = Sampler(G)
    n_timesteps = 10
    n_samples = 1000
    sampler.s0["∂(conflict)/∂t"] = 0.1
    sampler.sample_from_prior()
    sampler.calculate_log_prior()
    original_beta = sampler.A["∂(conflict)/∂t"]["food_security"]
    sampler.set_number_of_timesteps(n_timesteps)
    sampler.set_latent_state_sequence()
    sampler.sample_from_likelihood()
    sampler.calculate_log_likelihood()
    sampler.calculate_log_joint_probability()
    sampler.original_score = sampler.log_joint_probability
    map_estimate = original_beta
    map_log_joint_probability = sampler.log_joint_probability
    for i, _ in enumerate(trange(n_samples)):
        sampler.sample_from_posterior()
        if sampler.log_joint_probability > map_log_joint_probability:
            map_estimate = sampler.A["∂(conflict)/∂t"]["food_security"]
    assert map_estimate == approx(original_beta, abs=0.1 * abs(original_beta))
