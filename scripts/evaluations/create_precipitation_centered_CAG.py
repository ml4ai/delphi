import sys
import pickle
from delphi.export import to_agraph


def create_precipitation_centered_CAG(input, output, filename="CAG.pdf"):
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
    A = to_agraph(G)
    A.draw(filename, prog="dot")
    with open(output, "wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    create_precipitation_centered_CAG(sys.argv[1], sys.argv[2])
