import delphi.plotter as dp
from delphi.cpp.DelphiPython import AnalysisGraph, InitialBeta, InitialDerivative
import pandas as pd
import numpy as np

df_temp = pd.read_csv('../data/mini_use_case/TerraClimateOromiaMonthlyMaxTemp.csv')
temperature = df_temp['(deg C) Max Temperature (TerraClimate) at State, 1958-01-01 to 2019-12-31'].tolist()[:72]

df_rain = pd.read_csv('../data/mini_use_case/TerraClimateOromiaMontlhyPrecip.csv')
rain = df_rain['(mm) Precipitation (TerraClimate) at State, 1958-01-01 to 2019-12-31'].tolist()[:72]
# print(len(temperature)/12)
# print(len(rain))
# exit()
# rain = [56.49,35.32,31.59,16.34,7.52,16.09,74.37,80.28,23.24,11.45,5.76,9.6,31.96,32.4,20.02,14.89,7.67,5.06,67.74,84.23,35.63,8.66,7.37,14.24]
# temperature = [31.0215,29.8736,33.6469,35.6415,38.5646,38.3929,37.0828,35.7634,36.3652,34.8636,33.1313,31.7197,32.1485,31.574,33.1953,36.5846,38.1069,39.3582,36.7342,35.3044,35.1285,34.9057,32.3577,30.9242]
# exit()
# rain = [56.49,35.32,31.59,16.34,7.52,16.09,74.37,80.28,23.24,11.45,5.76,9.6,31.96,32.4,20.02,14.89,7.67,5.06,67.74,84.23,35.63,8.66,7.37,14.24]
# temperature = [31.0215,29.8736,33.6469,35.6415,38.5646,38.3929,37.0828,35.7634,36.3652,34.8636,33.1313,31.7197,32.1485,31.574,33.1953,36.5846,38.1069,39.3582,36.7342,35.3044,35.1285,34.9057,32.3577,30.9242]

rain = [0.067287097,0.948775,2.294458065,0.980603333,0.75403871,0.243053333,4.698829032,5.648793548,1.395853333,1.71013871,0.088853333,0.612916129,0.628951613,1.117355172,0.422251613,1.00248,1.226922581,0.945253333,3.664290323,4.430374194,1.35189,0.30026129,0.17042,0.156770968,0.262951613,0.3155,0.224541935,3.905033333,3.655470968,1.267846667,6.514112903,4.001903226,2.371223333,1.719193548,0.4052,0.097754839,1.29E-05,0.056485714,0.612593548,1.946536667,1.392032258,1.81064,5.636487097,6.073612903,3.73836,0.189503226,2.109933333,0.041948387,0.002009677,1.537889286,3.029506452,2.414076667,1.838977419,0.624603333,5.903183871,4.708770968,1.73639,1.50136129,0.132553333,1.126558065,0.711512903,0.056417241,3.617506452,2.823293333,2.162712903,1.488246667,5.011274194,6.265332258,1.59045,0.717329032,0.84738,0.038245161,0.312635484,0.008214286,1.990893548,1.447356667,0.611283871,2.711343333,6.126045161,5.360922581,1.009006667,3.353306452,0.68993,0.034229032,0.638777419,0.908292857,0.740129032,0.996463333,1.060212903,0.94176,7.602393548,6.537809677,2.832743333,1.914867742,0.08192,0.002280645,0.542119355,0.043289286,6.901003226,0.370483333,0.326690323,0.971193333,7.747980645,9.463974194,1.966903333,2.199083871,0.10037,0.059132258,0.002958065,0.003248276,0.182054839,0.65317,0.284403226,0.405456667,1.57133871,2.516,1.186123333,0.418283871,0.174276667,0.083293548,0.029319355,0.072082143,1.799454839,0.45108,0.947225806,1.14663,1.599316129,1.979583871,1.509206667,0.456532258,0.119636667,0.097506452,0.390154839,0.015535714,1.517209677,0.810713333,0.321122581,1.09399,2.976754839,1.879745161,1.102033333,0.085403226,0.021313333,0.423958065,0.0621,0.240214286,0.403612903,1.279313333,0.343429032,0.56549,1.357477419,2.878167742,1.3381,0.021722581,0.056593333,1.381941935,0.15293871,0.033848276,0.538754839,1.710623333,0.1561,0.735653333,1.548219355,1.287945161,0.896016667,0.3212,0.25264,1.269251613,0.529767742,0.020303571,0.681948387,5.66678,1.593519355,0.13908,1.427009677,1.426158065,0.819763333,0.111703226,0.162976667,3.55E-05,0.282587097,0.500996429,2.157316129,2.88453,0.847916129,0.825076667,5.779832258,5.873341935,1.80166,0.798580645,0.06097,0.638312903,0.296532258,0.759296429,0.48183871,2.13564,0.914458065,0.57174,4.016754839,2.676712903,1.53357,0.409864516,0.087673333,0.001303226,0.277087097,0.057248276,0.004441935,1.30485,0.584112903,0.351056667,3.051477419,3.621367742,1.480323333,0.256870968,0.91209,0.016370968,0.349096774,0.1044,0.534858065,0.897976667,0.104816129,0.739023333,4.489709677,4.163751613,0.888566667,1.398909677,0.04606,0.3052,0.00486129,1.65225,1.272480645,1.738746667,1.389903226,0.975606667,5.156887097,6.972422581,2.128933333,0.132574194,0.236806667,0.039932258,0.066177419,0.082425,1.075435484,0.657353333,2.481612903,0.225613333,1.453916129,1.706332258,0.67557,0.111554839,0.222553333,0.007632258,0.010341935,6.55E-05,0.146119355,1.21862,0.259967742,0.327793333,3.230758065,2.933706452,1.04061,0.0398,0.02899,0.076187097,0.345690323,0.017175,1.094945161,1.270896667,0.611083871,0.849813333,5.498580645,5.575745161,1.371143333,2.399748387,0.317456667,0.066935484,0.028819355,0.344992857,0.900948387,1.972063333,1.328677419,0.826326667,5.062932258,3.740464516,2.73512,1.170358065,0.141356667,0.082835484,0.284196774,0.030460714,0.380816129,0.023336667,0.799780645,1.522456667,0.990932258,5.702148387,0.964853333,0.196503226,0.575166667,0.818680645,0.259170968,0.318703448,0.818729032,4.820703333,2.27223871,0.494396667,5.14783871,4.638109677,1.609366667,0.295651613,0.161223333,0.043764516,0.001512903,1.785742857,1.725329032,1.322443333,2.432280645,0.32378,2.005419355,4.148558065,2.438363333,0.719467742,0.133733333,0.004945161,0.18523871,0.624142857,0.395341935,2.6592,0.176603226,0.471596667,2.196245161,3.461706452,3.37334,0.965896774,0.785046667,0.029367742,4.52E-05,1.35735,0.978096774,2.444236667,0.311396774,1.11368,0.597845161,2.926635484,5.163923333,1.876280645,0.98977,0.816735484,0.311477419,0.323831034,1.214196774,3.587106667,2.237845161,0.983026667,3.965074194,4.346283871,2.31646]
temperature = [24.86295806,25.13280714,26.54966129,29.79424333,30.95234516,32.11392667,31.3133871,30.89469355,30.93339333,28.55438065,25.50030667,24.46148387,23.27382903,22.39718276,26.38620645,28.46132,31.04445161,31.93411,31.08583548,29.0518871,29.22651333,27.91463871,25.62283333,24.27758387,22.73682258,22.71873571,26.90440645,27.09099333,29.07612903,31.63087333,31.36512581,31.18212581,30.76640333,28.82564839,26.51419667,24.70176129,24.26064516,25.4456,26.72726774,30.16119667,31.55742258,32.18229667,30.27816774,29.77531613,29.76672,28.68954194,26.45734333,24.03178387,24.27689355,25.49820357,26.19175484,28.42569333,31.27456452,32.53478,30.348,29.63275161,30.668,29.14517097,26.29693,25.11856129,24.68018387,26.1728931,27.17660968,28.99057333,29.39485161,31.00897667,31.11190968,30.14581613,29.83478667,28.72504516,25.64455667,24.22446774,23.39774839,24.49352143,26.59564839,28.20723333,31.10853871,32.19592333,31.45016452,31.03982903,31.70886333,27.52415484,26.21632667,25.16447419,24.46216129,25.575925,27.47573871,30.48144,31.88712581,32.94343,31.32420968,29.28708065,29.52118,28.83919355,26.52327667,24.66836452,24.45384516,26.67338571,26.1650871,30.09454,32.29913871,32.17986,30.45177742,30.24866452,30.46502667,27.24491935,26.15418667,24.47020645,24.13077419,25.69715517,27.2236871,29.86323333,30.83724516,31.83559,31.25467742,29.09985161,30.20353667,27.7406,25.47530667,23.6949129,22.95638387,24.76266071,26.65235161,29.95412,31.2869,32.02169333,30.61878065,28.99470645,30.00396,29.5072871,26.20904,24.8518129,23.5442129,25.41718929,27.52885161,28.54994333,31.47550323,32.14381667,32.49455806,31.33065484,30.03016333,29.42600323,26.98337,24.63960645,23.60346129,26.38779643,27.43636774,28.45589333,32.04939677,32.00567,31.26214516,28.87483548,30.23267667,28.85755484,26.54720667,23.94398387,24.50213871,24.27118966,26.94359032,27.94561667,31.54308065,31.89315333,31.87457419,31.04750323,31.08214667,28.86416774,26.74078,24.75294194,23.88877097,26.06273929,27.63122258,29.08497667,30.10806129,32.30074333,30.92069032,31.50618065,30.95165667,29.32583226,26.60961,24.46932581,24.27066129,26.245375,27.30174516,28.40079333,31.40136774,32.36381,31.09376774,30.1714,30.03189333,28.98043548,26.61085667,24.75597419,23.58546129,26.23269286,27.6941129,29.27583,32.08432903,31.97184,30.09358387,30.30682581,30.55679667,28.43743871,26.18127667,24.15435806,24.36885806,24.34289655,27.18429032,29.57276333,31.07354516,31.72307667,31.77259677,31.32441613,30.67252,28.38856129,25.18536667,23.83597742,24.08534194,26.39980714,27.70489355,29.89802333,31.28515806,32.71743333,31.9402129,31.10592903,31.31495333,28.483,26.61731,25.87396452,24.58355484,25.473925,27.04503871,29.63851,31.31575161,32.66308667,30.27800968,30.22356774,30.03330333,29.46778387,26.37985333,24.01269032,24.06503226,25.47452857,26.54421935,29.92544667,30.09432258,31.82947333,31.51112903,29.2929871,30.50952667,28.42188387,25.93507667,23.82311935,24.24770323,25.21965172,26.83623871,29.19745333,31.42051935,32.43823667,31.19894839,31.01210323,31.19151,28.78804839,26.86237667,25.50248065,24.83181613,25.6823,27.39812581,29.04286333,31.74748065,32.39842,31.14796774,29.03381613,30.91547333,28.72963548,26.70407,23.79045484,24.47675161,25.37844643,27.50585161,30.16779333,30.55179032,32.35886667,31.72154839,30.2342129,30.14706333,27.78348065,26.77146667,24.44642903,23.62613548,25.65624643,28.21600323,29.89016,31.10255484,32.18640333,33.03586774,32.23883548,31.72547333,30.40855484,27.57449667,25.3426129,25.09007742,26.18275517,30.29951613,28.54819667,31.26419032,33.11608,30.93897097,30.09242581,31.71994333,29.91460645,26.72585667,25.56935806,24.87566129,24.98533929,28.37990323,30.58911333,30.7755871,33.38314667,33.10136452,31.33474839,31.38021667,29.83076129,27.19399333,23.96029677,23.52738065,26.36581071,28.2635129,29.41648,31.54575806,32.58377333,31.64165806,31.09654516,31.05386,28.96214516,27.13166333,25.68454516,25.28315806,26.65149643,28.37367419,30.04280333,31.15893548,32.30068,32.47812903,30.34747742,31.06539667,28.27787419,27.33406,25.7966129,24.14383548,26.03038966,27.92293548,29.40862,31.62165806,32.35247333,32.12196667]

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
    # plt.plot(means)
    # plt.plot(std)
    # plt.show()

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
    G.set_random_seed(81)

    draw_CAG(G, 'modeling_efforts_CAG.png')

    print('\nTraining Model')
    G.run_train_model(res=200,
                      burn=1000,
                      initial_beta=InitialBeta.ZERO,
                      initial_derivative=InitialDerivative.DERI_ZERO,
                      use_heuristic=False,
                      use_continuous=False,
                      train_start_timestep=0,
                      train_timesteps=-1,#48, #12 * 52, #
                      concept_periods={'rain': 12},
                      concept_center_measures={'rain': "median"},  # mean, median
                      concept_models={'rain': "center"},  # center, absolute_change, relative_change
                      #concept_min_vals={'rain': 0}
                      #ext_concepts={'rain': get_rain_derivative}
                      #ext_concepts={'test': lambda x, s : x*s }
                      )

    # try:
    G.generate_prediction(325, 24) # 333 for last 24 months. 325 for two last complete years
    # G.generate_prediction(49, 23, constraints={12: [('rain', 'Monthly Precipitation (mm)', 120)]})
    # G.generate_prediction(12 * 52 + 1, 119)
    # except AnalysisGraph.BadCausemosInputException as e:
    #     print(e)
    #     exit()

    print('\n\nPlotting \n')
    model_state = G.get_complete_state()

    concept_indicators, edges, adjectives, polarities, edge_data, derivatives, data_range, data_set, pred_range, predictions, cis, log_likelihoods  = model_state

    print(data_range)
    print(pred_range[1:])

    dp.delphi_plotter(model_state, num_bins=400, rotation=90,
            out_dir='plots', file_name_prefix='me', save_csv=False)