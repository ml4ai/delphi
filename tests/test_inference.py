from conftest import *
from delphi.inference import *
from delphi.execution import construct_default_initial_state
from tqdm import trange


@pytest.fixture()
def A(G):
    return sample_transition_matrix_from_gradable_adjective_prior(G)


@pytest.fixture()
def s0(G):
    return construct_default_initial_state(G)


@pytest.fixture()
def latent_states(A, s0):
    return get_sequence_of_latent_states(A, s0, 3)


def test_sample_transition_matrix_from_gradable_adjective_prior(A):
    assert A.shape == (4, 4)
    for n in range(4):
        assert A.values[n][n] == 1.0


def test_get_sequence_of_latent_states(latent_states):
    for latent_state in latent_states:
        assert list(latent_state.values) == [1.0, 0.0, 1.0, 0.0]


@pytest.fixture()
def observed_state(G):
    return create_observed_state(G)


def test_create_observed_state(observed_state):
    assert observed_state == {
        "conflict": {
            "Prevalence of severe food insecurity in the total population Value": None
        },
        "food_security": {
            "Number of severely food insecure people Value": None
        },
    }


def test_sample_observed_state(G, s0):
    sampled_observed_state = sample_observed_state(G, s0)
    assert True




@pytest.fixture
def observed_states():
    conflict_indicator_values = [1.0, 2.0, 3.0]
    food_security_indicator_values = [0.1, 0.2, 0.3]
    return [
        {
            "conflict": {
                "Prevalence of severe food insecurity in the total population Value": food_security_indicator_value
            },
            "food_security": {
                "Number of severely food insecure people Value": food_security_indicator_value
            },
        }
        for conflict_indicator_value, food_security_indicator_value in zip(
            conflict_indicator_values, food_security_indicator_values
        )
    ]


def test_sampler(G, observed_states):
    sampler = Sampler(G, observed_states)
    for n in trange(1000):
        sample = sampler.get_sample()
    assert True
