import sys
import pickle

def create_quantified_CAG(pickleFile):
    with open(pickleFile, "rb") as f:
        G = pickle.load(f)

    G.res = 500
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()

if __name__ == "__main__":
    create_quantified_CAG(sys.argv[1])
