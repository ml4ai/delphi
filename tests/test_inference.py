from conftest import *
from delphi.inference import *
from delphi.execution import construct_default_initial_state


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


def test_evaluate_prior_pdf(A, G):
    """ Smokescreen test for evaluating prior pdf value (modulo normalization). """
    prior_pdf_value = evaluate_prior_pdf(A, G)

    # Assert true if the previous line runs without errors.
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


def test_evaluate_likelihood_pdf(G, latent_states, observed_states):
    """ Smokescreen test for evaluating likelihood pdf value (modulo normalization). """
    likelihood_pdf_value = evaluate_likelihood_pdf(
        G, latent_states, observed_states
    )
    assert True
