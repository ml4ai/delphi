from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from indra.statements import Concept, Influence, Evidence, Event, QualitativeDelta
conflict_string = "UN/events/human/conflict"
food_security_string = "UN/entities/human/food/food_security"
from delphi.AnalysisGraph import AnalysisGraph
from delphi.utils.indra import get_valid_statements_for_modeling

conflict_string = "UN/events/human/conflict"
human_migration_string = "UN/events/human/human_migration"
food_security_string = "UN/entities/human/food/food_security"

conflict = Event(
    Concept(
        conflict_string,
        db_refs={
            "TEXT": "conflict",
            "UN": [(conflict_string, 0.8), ("UN/events/crisis", 0.4)],
        },
    ),
    delta=QualitativeDelta(1, ["large"]),
)

food_security = Event(
    Concept(
        food_security_string,
        db_refs={"TEXT": "food security", "UN": [(food_security_string, 0.8)]},
    ),
    delta=QualitativeDelta(-1, ["small"]),
)

precipitation = Event(Concept("precipitation"))
human_migration = Event(
    Concept(
        human_migration_string,
        db_refs={"TEXT": "migration", "UN": [(human_migration_string, 0.8)]},
    ),
    delta=QualitativeDelta(1, ["large"]),
)


flooding = Event(Concept("flooding"))

s1 = Influence(
    conflict,
    food_security,
    evidence=Evidence(
        annotations={"subj_adjectives": ["large"], "obj_adjectives": ["small"]}
    ),
)

default_annotations = {"subj_adjectives": [], "obj_adjectives": []}

s2 = Influence(
    precipitation,
    food_security,
    evidence=Evidence(annotations=default_annotations),
)
s3 = Influence(
    precipitation, flooding, evidence=Evidence(annotations=default_annotations)
)
s4 = Influence(conflict, human_migration, evidence=Evidence(annotations =
    {"subj_adjectives": ["large"], "obj_adjectives": ["small"]}))

STS = [s1, s2, s3, s4]

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
    G.sample_from_likelihood(n_timesteps=10)
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
