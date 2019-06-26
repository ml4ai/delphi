import sys
import pickle


def create_parameterized_CAG(input, output, filename="CAG_with_indicators_and_values.pdf"):
    """ Create a CAG with mapped and parameterized indicators """
    with open(input, "rb") as f:
        G = pickle.load(f)
    G.parameterize(year=2017, month=4)
    G.get_timeseries_values_for_indicators()
    with open(output, "wb") as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    create_parameterized_CAG(sys.argv[1], sys.argv[3])
