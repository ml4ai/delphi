from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from indra.statements import Concept, Influence, Evidence, Event, QualitativeDelta
conflict_string = "UN/events/human/conflict"
food_security_string = "UN/entities/human/food/food_security"
from delphi.AnalysisGraph import AnalysisGraph
from delphi.utils.indra import get_valid_statements_for_modeling

concepts = {
    "conflict": {
        "grounding": "UN/events/human/conflict",
        "delta": {"polarity": 1, "adjective": ["large"]},
    },
    "food security": {
        "grounding": "UN/entities/human/food/food_security",
        "delta": {"polarity": -1, "adjective": ["small"]},
    },
    "migration": {
        "grounding": "UN/events/human/human_migration",
        "delta": {"polarity": 1, "adjective": ['small']},
    },
    "product": {
        "grounding": "UN/entities/natural/crop_technology/product",
        "delta": {"polarity": 1, "adjective": ['large']},
    },
}


def make_event(concept, attrs):
    return Event(
        Concept(
            attrs["grounding"],
            db_refs={"TEXT": concept, "UN": [(attrs["grounding"], 0.8)]},
        ),
        delta=QualitativeDelta(
            attrs["delta"]["polarity"], attrs["delta"]["adjective"]
        ),
    )


def make_statement(event1, event2):
    return Influence(
        event1,
        event2,
        evidence=Evidence(
            annotations={
                "subj_adjectives": event1.delta.adjectives,
                "obj_adjectives": event2.delta.adjectives,
            }
        ),
    )

events = {
    concept: make_event(concept, attrs) for concept, attrs in concepts.items()
}

s1 = make_statement(events["conflict"], events["food security"])
s2 = make_statement(events["migration"], events["product"])

STS = [s1, s2]

G = AnalysisGraph.from_statements(get_valid_statements_for_modeling(STS))
G.res=1000
G.sample_from_prior()
G.map_concepts_to_indicators()
G.parameterize(year=2014, month=12)

def test_inference_with_synthetic_data(G):
    """ Smokescreen test for sampler. """
    # Generate synthetic data

    # Sample a transition matrix from the prior
    A = G.sample_from_prior()[0]

    # Get the original value of our parameter of interest (the ground truth
    # value that we can use to evaluate our inference.
    original_beta = A[f"∂({conflict_string})/∂t"][f"{food_security_string}"]
    print(A.values)
    fig, ax = plt.subplots()
    sns.distplot(G.edges[conflict_string,
        food_security_string]["βs"], ax=ax)
    plt.savefig("betas_dist.pdf")

    # Initialize the latent state vector at time 0
    G.s0 = G.construct_default_initial_state()
    G.s0[f"∂({conflict_string})/∂t"] = 0.1

    # Given the initial latent state vector and the sampled transition matrix,
    # sample a sequence of latent states and observed states
    G.sample_from_likelihood(n_timesteps=5)
    G.latent_state_sequence = G.latent_state_sequences[0]
    G.observed_state_sequence = G.observed_state_sequences[0]

    # Save the current MAP estimate of the beta parameter (i.e. the value that
    # we are starting with.
    map_estimate = A[f"∂({conflict_string})/∂t"][food_security_string]


    # Create empty lists to hold the scores
    scores = []
    betas = []
    log_likelihoods = []
    log_priors = []
    map_estimates=[]


    for edge in G.edges(data=True):
        rand = 0.1*np.random.rand()
        A[f"∂({edge[0]})/∂t"][edge[1]] = rand

    n_samples: int = 1000
    map_estimate_matrix=A.copy()
    for i, _ in enumerate(trange(n_samples)):
        A = G.sample_from_posterior(A)
        betas.append(A[f"∂({conflict_string})/∂t"][food_security_string])

    plt.style.use("ggplot")
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(betas[::10], label="beta")
    axes[0].plot(
        original_beta * np.ones(len(betas[::10])), label="original_beta"
    )
    for ax in axes:
        ax.legend()
    plt.savefig("mcmc_results.pdf")
    fig, ax = plt.subplots()
    plt.hist(betas, bins=40)
    plt.savefig("betas.pdf")

    # This tolerance seems to work for now, so I'm leaving it in.
    print(map_estimate_matrix.values)
    # assert map_estimate == approx(original_beta, abs=0.1)

test_inference_with_synthetic_data(G)
