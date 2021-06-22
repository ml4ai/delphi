import delphi.plotter as dp
from delphi.cpp.DelphiPython import AnalysisGraph, InitialBeta, InitialDerivative
import pandas as pd
import numpy as np

df_temp = pd.read_csv('../data/mini_use_case/TerraClimateOromiaMonthlyMaxTemp.csv')
temperature = df_temp['(deg C) Max Temperature (TerraClimate) at State, 1958-01-01 to 2019-12-31'].tolist()[:72]

df_rain = pd.read_csv('../data/mini_use_case/TerraClimateOromiaMontlhyPrecip.csv')
rain = df_rain['(mm) Precipitation (TerraClimate) at State, 1958-01-01 to 2019-12-31'].tolist()[:72]
print(len(temperature))
print(len(rain))
# exit()
# rain = [56.49,35.32,31.59,16.34,7.52,16.09,74.37,80.28,23.24,11.45,5.76,9.6,31.96,32.4,20.02,14.89,7.67,5.06,67.74,84.23,35.63,8.66,7.37,14.24]
# temperature = [31.0215,29.8736,33.6469,35.6415,38.5646,38.3929,37.0828,35.7634,36.3652,34.8636,33.1313,31.7197,32.1485,31.574,33.1953,36.5846,38.1069,39.3582,36.7342,35.3044,35.1285,34.9057,32.3577,30.9242]
# exit()
# rain = [56.49,35.32,31.59,16.34,7.52,16.09,74.37,80.28,23.24,11.45,5.76,9.6,31.96,32.4,20.02,14.89,7.67,5.06,67.74,84.23,35.63,8.66,7.37,14.24]
# temperature = [31.0215,29.8736,33.6469,35.6415,38.5646,38.3929,37.0828,35.7634,36.3652,34.8636,33.1313,31.7197,32.1485,31.574,33.1953,36.5846,38.1069,39.3582,36.7342,35.3044,35.1285,34.9057,32.3577,30.9242]

def compute_partitioned_mean_std(data, period):
    partitions = {}
    means = []
    std = []

    for partition in range(period):
        partitions[partition] = []
        std.append(0)
        means.append(0)

    for idx, val in enumerate(data):
        partitions[idx % period].append(val)

    for partition, vals in partitions.items():
        means[partition] = np.median(vals) #sum(vals) / len(vals)
        std[partition] = np.std(vals)

    print(partitions)
    print(means)
    print(std)
    plt.plot(means)
    plt.plot(std)
    plt.show()

def create_base_CAG(kde_kernels=4):
    statements = [
        (
            ("", 1, "rain"),
            ("", -1, "temperature"),
        )
    ]

    data = {"rain": ("Monthly Precipitation (mm)", rain),
            "temperature": ("Monthly Max Temperature (F)", temperature)
            }
    G = AnalysisGraph.from_causal_fragments_with_data((statements, data), kde_kernels)
    return G


def get_rain_derivative(ts, sf):
    #print(ts)
    return (rain[ts + 1] - rain[ts]) / sf

def draw_CAG(G, file_name):
    G.to_png(
        file_name,
        rankdir="TB",
        simplified_labels=False,
    )


if __name__ == "__main__":
    json_inputs = [
        ["../tests/data/delphi/create_model_test.json",                 # 0. Missing data and mixed sampling frequency
         "../tests/data/delphi/experiments_projection_test.json"],
        ["../tests/data/delphi/create_model_ideal.json",                # 1. Ideal data with gaps of 1
         "../tests/data/delphi/experiments_projection_ideal.json"],
        ["../tests/data/delphi/create_model_input_2.json",              # 2. Usual Data
         "../tests/data/delphi/experiments_projection_input_2.json"],
        ["../tests/data/delphi/create_model_ideal_10.json",             # 3. Ideal data with gaps of 10
         "../tests/data/delphi/experiments_projection_ideal_2.json"],
        ["../tests/data/delphi/create_model_ideal_3.json",              # 4. Ideal data with real epochs
         "../tests/data/delphi/experiments_projection_ideal_3.json"],
        ["../tests/data/delphi/create_model_input_2_no_data.json",      # 5. No data
         "../tests/data/delphi/experiments_projection_input_2.json"],
        ["../tests/data/delphi/create_model_input_2_partial_data.json", # 6. Partial data
         "../tests/data/delphi/experiments_projection_input_2.json"],
        ["../tests/data/delphi/create_model_input_new.json",            # 7. Updated create model format
         ""],
        ["../tests/data/delphi/causemos_create.json",                   # 8. Oldest test data
         "../tests/data/delphi/causemos_experiments_projection_input.json"],
    ]

    input_idx = 2
    causemos_create_model = json_inputs[input_idx][0]
    causemos_create_experiment = json_inputs[input_idx][1]

    # G = create_base_CAG(causemos_create_model)
    G = create_base_CAG(kde_kernels=1000)
    #G = create_base_CAG('', 100)

    draw_CAG(G, 'modeling_efforts_CAG.png')


    print('\nTraining Model')
    G.run_train_model(res=200,
                      burn=10000,
                      initial_beta=InitialBeta.ZERO,
                      initial_derivative=InitialDerivative.DERI_ZERO,
                      use_heuristic=False,
                      use_continuous=False,
                      train_start_timestep=0,
                      train_timesteps=48,
                      concept_periods={'rain': 12},
                      concept_center_measures={'rain': "median"},  # mean, median
                      concept_models={'rain': "center"},  # center, absolute_change, relative_change
                      #concept_min_vals={'rain': 0}
                      #ext_concepts={'rain': get_rain_derivative}
                      #ext_concepts={'test': lambda x, s : x*s }
                      )

    # try:
    # G.generate_prediction(49, 24)
    G.generate_prediction(49, 23, constraints={12: [('rain', 'Monthly Precipitation (mm)', 120)]})
    # except AnalysisGraph.BadCausemosInputException as e:
    #     print(e)
    #     exit()

    print('\n\nPlotting \n')
    model_state = G.get_complete_state()

    concept_indicators, edges, adjectives, polarities, edge_data, derivatives, data_range, data_set, pred_range, predictions, cis  = model_state

    print(data_range)
    print(pred_range[1:])

    dp.delphi_plotter(model_state, num_bins=400, rotation=45,
            out_dir='plots', file_name_prefix='', save_csv=False)
