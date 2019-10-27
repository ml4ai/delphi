import delphi.evaluation as EN
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
    set_indicator(
        G,
        "wm/concept/indicator_and_reported_property/weather/rainfall",
        "Average Precipitation",
        "DSSAT",
    )

    set_indicator(
        G,
        "wm/concept/indicator_and_reported_property/agriculture/Crop_Production",
        "Average Harvested Weight at Maturity (Maize)",
        "DSSAT",
    )

    set_indicator(
        G,
        "wm/concept/causal_factor/condition/food_insecurity",
        "IPC Phase Classification",
        "FEWSNET",
    )

    set_indicator(
        G,
        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/price_or_cost/food_price",
        "Consumer price index",
        "WDI",
    )

    set_indicator(
        G,
        "wm/concept/indicator_and_reported_property/conflict/population_displacement",
        "Internally displaced persons, total displaced by conflict and violence",
        "WDI",
    )

    set_indicator(
        G,
        "wm/concept/causal_factor/condition/tension",
        "Conflict incidences",
        "None",
    )


def draw_CAG(G):
    G.to_png(
        "Oct2019EvalCAG.png",
        rankdir="TB",
        node_to_highlight="wm/concept/causal_factor/condition/food_insecurity",
        simplified_labels=False,
    )


if __name__ == "__main__":
    G = create_base_CAG("data/Model4.json")
    G.set_random_seed(2018)
    curate_indicators(G)
    G.data_heuristic = False
    G.parameterize("South Sudan", "Jonglei", "", 2017, 4, {})
    ind = G[
        "wm/concept/indicator_and_reported_property/agriculture/Crop_Production"
    ].get_indicator("Average Harvested Weight at Maturity (Maize)")

    draw_CAG(G)
    G.train_model(
        2014,
        6,
        2016,
        3,
        country="South Sudan",
        res=200,
        burn=10000,
        use_heuristic=False,
    )
    G.set_default_initial_state()
    G.s0[1] = 0.1
    preds = G.generate_prediction(2016, 3, 2016, 7)
    # preds = G.generate_prediction(2018, 1, 2018, 2)
    EN.pred_plot(
        preds,
        "IPC Phase Classification",
        0.95,
        plot_type="Prediction",
        show_rmse=True,
        show_training_data=True,
        use_heuristic_for_true=False,
        save_as="Oct2019EvalPred.png",
    )
