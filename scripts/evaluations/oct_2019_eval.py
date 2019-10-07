from delphi.cpp.DelphiPython import AnalysisGraph

G = AnalysisGraph.from_uncharted_json_file("data/Model4.json")
G.merge_nodes(
    "wm/concept/causal_factor/condition/food_security",
    "wm/concept/causal_factor/condition/food_insecurity",
    same_polarity=False,
)
G = G.get_subgraph_for_concept(
    "wm/concept/causal_factor/condition/food_insecurity", inward=True
)
G.map_concepts_to_indicators()

def set_indicator(G, concept, indicator_new, source):
    G.delete_all_indicators(concept)
    G.set_indicator(concept, indicator_new, source)

set_indicator(G, "wm/concept/indicator_and_reported_property/weather/rainfall",
        "Average Precipitation", "DSSAT")

set_indicator(G,
    "wm/concept/indicator_and_reported_property/agriculture/Crop_Production",
    "Average Harvested Weight at Maturity (Maize)",
    "DSSAT",
)

set_indicator(G,
    "wm/concept/causal_factor/condition/food_insecurity",
    "IPC Phase Classification",
    "FEWSNET",
)


G.to_png(
    "Oct2019EvalCAG.png",
    rankdir="TB",
    node_to_highlight="wm/concept/causal_factor/condition/food_insecurity",
)

print("Training model...")
G.train_model(country="South Sudan")
