from delphi.cpp.DelphiPython import AnalysisGraph

def create_base_CAG(uncharted_json_file):
    G = AnalysisGraph.from_uncharted_json_file(uncharted_json_file)
    G.merge_nodes(
        "wm/concept/causal_factor/condition/food_security",
        "wm/concept/causal_factor/condition/food_insecurity",
        same_polarity=False,
    )
    G = G.get_subgraph_for_concept(
        "wm/concept/causal_factor/condition/food_insecurity", inward=True
    )
    G.map_concepts_to_indicators()
    return G

def set_indicator(G, concept, indicator_new, source):
    G.delete_all_indicators(concept)
    G.set_indicator(concept, indicator_new, source)

def curate_indicators(G):
    # set_indicator(G, "wm/concept/indicator_and_reported_property/weather/rainfall",
            # "Average Precipitation", "DSSAT")

    # set_indicator(G,
        # "wm/concept/indicator_and_reported_property/agriculture/Crop_Production",
        # "Average Harvested Weight at Maturity (Maize)",
        # "DSSAT",
    # )

    # set_indicator(G,
        # "wm/concept/causal_factor/condition/food_insecurity",
        # "IPC Phase Classification",
        # "FEWSNET",
    # )
    pass


def draw_CAG(G):
    G.to_png(
        "Oct2019EvalCAG.png",
        rankdir="TB",
        node_to_highlight="wm/concept/causal_factor/condition/food_insecurity",
        simplified_labels=False
    )


if __name__ == "__main__":
    G = create_base_CAG("data/Model4.json")
    curate_indicators(G)
    draw_CAG(G)
    G.train_model(2014, 5, 2016, 4, country="South Sudan", burn=1)
