import delphi.plotter as dp
from delphi.cpp.DelphiPython import AnalysisGraph
import pandas as pd
import numpy as np


def set_indicator(G, concept, indicator_new, source):
    G.delete_all_indicators(concept)
    G.set_indicator(concept, indicator_new, source)

def remove_node(G, concept, indicator_new, source):
    G.delete_all_indicators(concept)
    G.remove_node(concept)


def curate_indicators(G):
    '''
    set_indicator(
        G,
        "wm/concept/indicator_and_reported_property/weather/rainfall",
        "Average Precipitation",
        "DSSAT",
    )
    '''

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

    '''
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
    #G.remove_node("wm/concept/indicator_and_reported_property/agriculture/Crop_Production")
    G.remove_node("wm/concept/indicator_and_reported_property/weather/rainfall")
    G.remove_node("wm/concept/indicator_and_reported_property/conflict/population_displacement")
    G.remove_node("wm/concept/causal_factor/condition/tension")
    '''

def create_base_CAG(causemos_create_model, res = 4):
    if causemos_create_model:
        G = AnalysisGraph.from_causemos_json_file(causemos_create_model, res);
    else:
        statements = [
            (
                ("large", 1, "wm/concept/indicator_and_reported_property/agriculture/Crop_Production"),
                ("small", -1, "wm/concept/causal_factor/condition/food_insecurity"),
            )
        ]
        G = AnalysisGraph.from_causal_fragments(statements)
        G.map_concepts_to_indicators()
        curate_indicators(G)
    return G


def draw_CAG(G, file_name):
    G.to_png(
        file_name,
        rankdir="TB",
        simplified_labels=False,
    )


if __name__ == "__main__":
    json_inputs = [
        ["../tests/data/delphi/create_model_test.json",                 # 0.
         "../tests/data/delphi/experiments_projection_test.json"],
        ["../tests/data/delphi/create_model_ideal.json",                # 1. Ideal data with gaps of 1
         "../tests/data/delphi/experiments_projection_ideal.json"],
        ["../tests/data/delphi/create_model_input_2.json",              # 2. Ideal data with gaps of 2
         "../tests/data/delphi/experiments_projection_input_2.json"],
        ["../tests/data/delphi/create_model_ideal_10.json",             # 3. Ideal data with gaps of 10
         "../tests/data/delphi/experiments_projection_ideal_2.json"],
        ["../tests/data/delphi/create_model_ideal_3.json",              # 4. Ideal data with real epochs
         "../tests/data/delphi/experiments_projection_ideal_3.json"],
        ["../tests/data/delphi/create_model_input_2_no_data.json",      # 5. No data
         "../tests/data/delphi/experiments_projection_input_2.json"],
        ["../tests/data/delphi/create_model_input_2_partial_data.json", # 6. Partial data
         "../tests/data/delphi/experiments_projection_input_2.json"],
        ["../tests/data/delphi/causemos_create-model_nodata.json",      # 7. No data another model
         "../tests/data/delphi/experiments_projection_input_2.json"],
        ["../tests/data/delphi/causemos_create-model.json",             # 8. Usual test data
         "../tests/data/delphi/causemos_experiments_projection_input.json"],
    ]

    input_idx = 0
    causemos_create_model = json_inputs[input_idx][0]
    causemos_create_experiment = json_inputs[input_idx][1]

    G = create_base_CAG(causemos_create_model, 1000)
    #G = create_base_CAG('', 100)

    draw_CAG(G, 'plot_testing_CAG.png')

    
    print('\nTraining Model')
    try:
        preds = G.run_causemos_projection_experiment_from_json_file(
                filename=causemos_create_experiment,
                burn=10,
                res=10)
    except AnalysisGraph.BadCausemosInputException as e:
        print(e)
        exit()

    print('\n\nPlotting \n')
    model_state = G.get_complete_state()

    concept_indicators, edges, adjectives, polarities, edge_data, derivatives, data_range, data_set, pred_range, predictions, cis  = model_state

    print(data_range)
    print(pred_range[1:])

    dp.delphi_plotter(model_state, num_bins=400, rotation=45,
            out_dir='plots', file_name_prefix='')
