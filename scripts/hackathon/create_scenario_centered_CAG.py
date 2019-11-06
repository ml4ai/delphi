import sys
import pickle
import random
import numpy as np
np.random.seed(87)
random.seed(87)

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
    A=G.to_agraph(nodes_to_highlight="UN/events/human/human_migration")
    A.draw("CAG.pdf", prog="dot")
    print(G.nodes)
    with open(output, "wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    create_scenario_centered_CAG(sys.argv[1], sys.argv[2])
