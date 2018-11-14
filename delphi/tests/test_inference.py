from delphi.tests.conftest import *
from delphi.inference import *
from delphi.execution import construct_default_initial_state

def test_sampler(G):
    A = sample_transition_matrix_from_gradable_adjective_prior(G)
    s0 = construct_default_initial_state(G)
    latent_states = get_sequence_of_latent_states(A, s0, 3)
    for n in G.nodes(data=True):
        print(n[1])
    assert True
