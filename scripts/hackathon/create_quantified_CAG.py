import sys
import pickle
from delphi import AnalysisGraph as AG

def create_quantified_CAG(input, output):
    with open(input, "rb") as f:
        G = pickle.load(f)

    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.get_timeseries_values_for_indicators()
    A = AG.to_agraph(G, filename="CAG.pdf")
    with open(output, "wb") as f:
        pickle.dump(G, f)
    A = AG.to_agraph(G, indicators=True)
    A.draw("CAG_with_indicators.pdf", prog="dot")

    A = AG.to_agraph(G, indicators=True, indicator_values=True)
    A.draw("CAG_with_indicators_and_values.pdf", prog="dot")

if __name__ == "__main__":
    create_quantified_CAG(sys.argv[1], sys.argv[2])
