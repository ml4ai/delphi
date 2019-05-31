import sys
import pickle


def create_scenario_centered_CAG(input, output):
    """ Get a CAG that examines the upstream effects of changes in human
    migration. """

    with open(input, "rb") as f:
        G = pickle.load(f)
    G = G.get_subgraph_for_concept(
        "UN/events/human/human_migration", depth=2, reverse=True
    )
    G.prune(cutoff=2)

    # Manually correcting a bad CWMS extraction
    #G.edges[
    #    "UN/events/weather/precipitation",
    #    "UN/entities/human/infrastructure/transportation/road",
   # ]["InfluenceStatements"][0].obj_delta["polarity"] = -1
    with open(output, "wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    create_scenario_centered_CAG(sys.argv[1], sys.argv[2])
