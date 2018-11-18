from conftest import *
from delphi.inference import *
from tqdm import trange


def test_sampler(G):
    """ Smokescreen test for sampler. """
    sampler = Sampler(G)
    n_timesteps = 2
    n_samples = 10000
    sampler.sample_from_prior()
    original_beta = sampler.A["∂(conflict)/∂t"]["food_security"]
    sampler.set_number_of_timesteps(n_timesteps)
    sampler.set_latent_state_sequence()
    sampler.sample_from_likelihood()
    samples = []
    for _ in trange(n_samples):
        sampler.sample_from_posterior()
        samples.append(sampler.A["∂(conflict)/∂t"]["food_security"])
    assert True
