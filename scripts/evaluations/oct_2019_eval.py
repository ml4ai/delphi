from delphi.cpp.DelphiPython import AnalysisGraph

G = AnalysisGraph.from_uncharted_json_file("data/Model3.json")
G.merge_nodes("wm/concept/causal_factor/condition/food_security",
        "wm/concept/causal_factor/condition/food_insecurity",
        same_polarity=False)
G.merge_nodes("wm/concept/causal_factor/economic_and_commerce/economic_activity/market/currency_devaluation",
        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/inflation",
        same_polarity=True)
G = G.get_subgraph_for_concept("wm/concept/causal_factor/condition/food_insecurity",
        inward=True)
G.remove_edge("wm/concept/causal_factor/crisis_and_disaster/crisis/environmental_factor/natural_disaster/drought",
        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/supply/food_supply")
G.map_concepts_to_indicators()
G.delete_all_indicators("wm/concept/causal_factor/condition/food_insecurity")
G.set_indicator("wm/concept/causal_factor/condition/food_insecurity", "IPC Phase Classification", "FEWSNET")
G.to_png("Oct2019EvalCAG.png", rankdir="TB", node_to_highlight="wm/concept/causal_factor/condition/food_insecurity")
# G.train_model()
