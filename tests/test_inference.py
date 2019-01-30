from conftest import G

def test_inference(G):
    G.res=3
    G.sample_from_prior()
    A = G.transition_matrix_collection[0].values
    s0 = G.construct_default_initial_state()
    s0[s0.index[1]] = 0.1
    G.sample_from_likelihood(s0)
    assert True
