import sys
import pickle


def create_precipitation_centered_CAG(input, output):
    """ Get a CAG that examines the downstream effects of changes in precipitation. """

    with open(input, "rb") as f:
        G = pickle.load(f)
    G = G.get_subgraph_for_concept(
        "UN/events/weather/precipitation", depth=2, reverse=False
    )
    G.prune(cutoff=2)

    # Manually correcting a bad CWMS extraction
    G.edges[
        "UN/events/weather/precipitation",
        "UN/entities/human/infrastructure/transportation/road",
    ]["InfluenceStatements"][0].obj_delta["polarity"] = -1
    with open(output, "wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    create_precipitation_centered_CAG(sys.argv[1], sys.argv[2])
