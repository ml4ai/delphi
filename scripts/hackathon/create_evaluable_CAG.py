import sys
import pickle

def create_evaluable_CAG(input, output):
    with open(input, "rb") as f:
        G = pickle.load(f)

    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    with open(output, "wb") as f:
        pickle.dump(G, f)
if __name__ == "__main__":
    create_evaluable_CAG(sys.argv[1], sys.argv[2])
