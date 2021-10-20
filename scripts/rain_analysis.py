import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})

out_dir = 'plots/rainfall_causemos/'
if out_dir:
    out_path = pathlib.Path(out_dir)
    if not out_path.is_dir():
        print(f'\nMaking output directory: {out_dir}')
        out_path.mkdir(parents=True, exist_ok=True)

month_names = {0: 'January',
               1: 'February',
               2: 'March',
               3: 'April',
               4: 'May',
               5: 'June',
               6: 'July',
               7: 'August',
               8: 'September',
               9: 'October',
               10: 'November',
               11: 'December'}

df_rain = pd.read_csv('../data/mini_use_case/TerraClimateOromiaMontlhyPrecip.csv')
rain = np.array(df_rain['(mm) Precipitation (TerraClimate) at State, 1958-01-01 to 2019-12-31'].tolist())

rain = [0.067287097,0.948775,2.294458065,0.980603333,0.75403871,0.243053333,4.698829032,5.648793548,1.395853333,1.71013871,0.088853333,0.612916129,0.628951613,1.117355172,0.422251613,1.00248,1.226922581,0.945253333,3.664290323,4.430374194,1.35189,0.30026129,0.17042,0.156770968,0.262951613,0.3155,0.224541935,3.905033333,3.655470968,1.267846667,6.514112903,4.001903226,2.371223333,1.719193548,0.4052,0.097754839,1.29E-05,0.056485714,0.612593548,1.946536667,1.392032258,1.81064,5.636487097,6.073612903,3.73836,0.189503226,2.109933333,0.041948387,0.002009677,1.537889286,3.029506452,2.414076667,1.838977419,0.624603333,5.903183871,4.708770968,1.73639,1.50136129,0.132553333,1.126558065,0.711512903,0.056417241,3.617506452,2.823293333,2.162712903,1.488246667,5.011274194,6.265332258,1.59045,0.717329032,0.84738,0.038245161,0.312635484,0.008214286,1.990893548,1.447356667,0.611283871,2.711343333,6.126045161,5.360922581,1.009006667,3.353306452,0.68993,0.034229032,0.638777419,0.908292857,0.740129032,0.996463333,1.060212903,0.94176,7.602393548,6.537809677,2.832743333,1.914867742,0.08192,0.002280645,0.542119355,0.043289286,6.901003226,0.370483333,0.326690323,0.971193333,7.747980645,9.463974194,1.966903333,2.199083871,0.10037,0.059132258,0.002958065,0.003248276,0.182054839,0.65317,0.284403226,0.405456667,1.57133871,2.516,1.186123333,0.418283871,0.174276667,0.083293548,0.029319355,0.072082143,1.799454839,0.45108,0.947225806,1.14663,1.599316129,1.979583871,1.509206667,0.456532258,0.119636667,0.097506452,0.390154839,0.015535714,1.517209677,0.810713333,0.321122581,1.09399,2.976754839,1.879745161,1.102033333,0.085403226,0.021313333,0.423958065,0.0621,0.240214286,0.403612903,1.279313333,0.343429032,0.56549,1.357477419,2.878167742,1.3381,0.021722581,0.056593333,1.381941935,0.15293871,0.033848276,0.538754839,1.710623333,0.1561,0.735653333,1.548219355,1.287945161,0.896016667,0.3212,0.25264,1.269251613,0.529767742,0.020303571,0.681948387,5.66678,1.593519355,0.13908,1.427009677,1.426158065,0.819763333,0.111703226,0.162976667,3.55E-05,0.282587097,0.500996429,2.157316129,2.88453,0.847916129,0.825076667,5.779832258,5.873341935,1.80166,0.798580645,0.06097,0.638312903,0.296532258,0.759296429,0.48183871,2.13564,0.914458065,0.57174,4.016754839,2.676712903,1.53357,0.409864516,0.087673333,0.001303226,0.277087097,0.057248276,0.004441935,1.30485,0.584112903,0.351056667,3.051477419,3.621367742,1.480323333,0.256870968,0.91209,0.016370968,0.349096774,0.1044,0.534858065,0.897976667,0.104816129,0.739023333,4.489709677,4.163751613,0.888566667,1.398909677,0.04606,0.3052,0.00486129,1.65225,1.272480645,1.738746667,1.389903226,0.975606667,5.156887097,6.972422581,2.128933333,0.132574194,0.236806667,0.039932258,0.066177419,0.082425,1.075435484,0.657353333,2.481612903,0.225613333,1.453916129,1.706332258,0.67557,0.111554839,0.222553333,0.007632258,0.010341935,6.55E-05,0.146119355,1.21862,0.259967742,0.327793333,3.230758065,2.933706452,1.04061,0.0398,0.02899,0.076187097,0.345690323,0.017175,1.094945161,1.270896667,0.611083871,0.849813333,5.498580645,5.575745161,1.371143333,2.399748387,0.317456667,0.066935484,0.028819355,0.344992857,0.900948387,1.972063333,1.328677419,0.826326667,5.062932258,3.740464516,2.73512,1.170358065,0.141356667,0.082835484,0.284196774,0.030460714,0.380816129,0.023336667,0.799780645,1.522456667,0.990932258,5.702148387,0.964853333,0.196503226,0.575166667,0.818680645,0.259170968,0.318703448,0.818729032,4.820703333,2.27223871,0.494396667,5.14783871,4.638109677,1.609366667,0.295651613,0.161223333,0.043764516,0.001512903,1.785742857,1.725329032,1.322443333,2.432280645,0.32378,2.005419355,4.148558065,2.438363333,0.719467742,0.133733333,0.004945161,0.18523871,0.624142857,0.395341935,2.6592,0.176603226,0.471596667,2.196245161,3.461706452,3.37334,0.965896774,0.785046667,0.029367742,4.52E-05,1.35735,0.978096774,2.444236667,0.311396774,1.11368,0.597845161,2.926635484,5.163923333,1.876280645,0.98977,0.816735484,0.311477419,0.323831034,1.214196774,3.587106667,2.237845161,0.983026667,3.965074194,4.346283871,2.31646]
rain = np.array(rain[:12*29])

print(len(rain))
print(len(rain)/12)
# df_rain.rename(columns={'(mm) Precipitation (TerraClimate) at State, 1958-01-01 to 2019-12-31': 'Rainfall (mm)'})
# df_rain['DateTime'] = df_rain['DateTime'].apply(lambda dt : pd.to_datetime(dt))
# df_rain['Month'] = df_rain['DateTime'].apply(lambda dt : dt.month)
# df_rain['Year'] = df_rain['DateTime'].apply(lambda dt : dt.year - 100)
# df_rain.to_csv('testrain.csv')
# print(df_rain.head())
# exit()

'''
absolute_change = {1, -0.374757, -0.0660294, -0.269959, -0.156134, 0.151708, 1.03169, 0.10462, -1.00974, -0.20871, ... size:48} (std::vector<double>)
mean_sequence = {1, 0.625243, 0.559214, 0.289255, 0.133121, 0.284829, 1.31652, 1.42114, 0.4114, 0.202691, ... size:48} (std::vector<double>)
relative_change = {-0.187378, -0.0406274, -0.173138, -0.121104, 0.133885, 0.802976, 0.0451628, -0.417051, -0.147874, -0.0837504, ... size:47} (std::vector<double>)
n.name = "rain" (std::string)
n.partitioned_data[0].second = {1, 0.565764, 0.322181, 0.482917} (std::vector<double>)
part_means_debug = {0.524341, 0.599398, 0.694282, 0.276421, 0.168879, 0.166755, 1.26615, 1.4561, 0.61294, 0.1718, ... size:12} (std::vector<double>)
n.absolute_change_medians = {0.00761197, -0.0660294, -0.18729, -0.141972, -0.00212427, 1.11869, 0.0455833, -0.935033, -0.519472, 0.0715171, ... size:12} (std::vector<double>)
n.relative_change_medians = {0.524341, 0.599398, 0.694282, 0.276421, 0.168879, 0.166755, 1.26615, 1.4561, 0.61294, 0.1718, ... size:12} (std::vector<double>)

from median = {0.524341, 0.599398, 0.694282, 0.276421, 0.168879, 0.166755, 1.26615, 1.4561, 0.61294, 0.1718, ... size:71} (std::vector<double>)
abs change = {0.524341, 0.531953, 0.465923, 0.278633, 0.136661, 0.134537, 1.25323, 1.29881, 0.363781, -0.155691, ... size:71} (std::vector<double>)
difference = {0, -0.0674456, -0.228359, 0.00221278, -0.0322181, -0.0322181, -0.0129226, -0.157284, -0.249159, -0.327492, ... size:71} (std::vector<double>)

'''
def changes_analysis(data, plot_no, modifire, period=12):
    scaling_factor = data[0]
    scaled_data = np.array(data) / scaling_factor
    absolute_change = np.diff(scaled_data)
    relative_change = absolute_change / (scaled_data[0: -1] + 1)

    # print(scaled_data[0:11])
    # print(absolute_change[0:11])
    # print(relative_change[0:11])

    p_data = {}
    p_absolute_change = {}
    p_relative_change = {}
    med_data = []
    mean_data = []
    med_absolute_change = []
    med_relative_change = []
    mean_absolute_change = []

    for partition in range(period):
        p_data[partition] = []
        p_absolute_change[partition] = []
        p_relative_change[partition] = []
        med_data.append(0)
        mean_data.append(0)
        med_absolute_change.append(0)
        med_relative_change.append(0)
        mean_absolute_change.append(0)

    for idx in range(len(scaled_data)):
        p_data[idx % period].append(scaled_data[idx])

        if idx < len(absolute_change):
            p_absolute_change[idx % period].append(absolute_change[idx])
            p_relative_change[idx % period].append(relative_change[idx])

    for partition in range(period):
        med_data[partition] = np.median(p_data[partition])
        mean_data[partition] = np.mean(p_data[partition])
        med_absolute_change[partition] = np.median(p_absolute_change[partition])
        med_relative_change[partition] = np.median(p_relative_change[partition])
        mean_absolute_change[partition] = np.mean(p_absolute_change[partition])

    med_absolute_change = np.array(med_absolute_change)
    med_absolute_change_0_centered = (med_absolute_change - np.mean(med_absolute_change)) * scaling_factor
    print(np.mean(med_absolute_change))
    print(np.mean(med_absolute_change_0_centered))
    print(list(med_absolute_change))
    print(list(med_absolute_change_0_centered))

    med_data = np.array(med_data)
    # mean_data = np.array(mean_data)
    # print(p_data[0], '\n')
    # print(scaling_factor, '\n')
    # print('Median :', med_data * scaling_factor, '\n')
    # print('Mean   :', mean_data * scaling_factor, '\n')
    # print('Diff   :', mean_data * scaling_factor - med_data * scaling_factor, '\n')
    # print(med_absolute_change, '\n')
    # print(med_relative_change, '\n')

    med_data = list(med_data)
    med_data.append(med_data[0])
    med_data = np.array(med_data)
    med_data_absolute_change = np.diff(med_data)
    med_data_relative_change = med_data_absolute_change / (med_data[0: -1] + 1)

    print('\nrel change\n', list(med_data_relative_change))

    # print(list(med_data), '\n')
    # print(list(med_data_absolute_change), '\n')
    #
    # med_data = np.array(med_data)
    # print(list(med_data[1:] - (med_data[:-1] + med_data_absolute_change)), '\n')
    #
    # pred = [med_data[0]]
    # for ts in range(period):
    #     partition = ts % period
    #     pred.append(pred[ts] + med_data_absolute_change[partition])
    #
    # # print(list(med_data[1:] - np.array(pred)), '\n')
    # print(list(med_data))
    # print(pred, '\n')
    # # exit()

    pred1 = [med_data[0]]
    pred2 = [med_data[0]]
    pred3 = [med_data[0]]
    pred4 = [med_data[0]]
    pred5 = [med_data[0]]
    pred6 = [med_data[0]]
    pred7 = [med_data[0]]
    # for ts in range(1, 48):
    for ts in range(1, len(scaled_data)):
        # for ts in range(1, 12):
        partition = (ts - 1) % period
        pred1.append(pred1[ts-1] + med_data_absolute_change[partition])
        pred2.append(pred2[ts-1] + med_absolute_change[partition])
        pred4.append(pred4[ts-1] + mean_absolute_change[partition])
        pred5.append(pred5[ts-1] + (pred5[ts-1] + 1) * med_relative_change[partition])
        pred6.append(pred2[ts-1] + med_absolute_change_0_centered[partition])
        pred7.append(pred7[ts-1] + (pred7[ts-1] + 1) * med_data_relative_change[partition])

        pred3.append(med_data[ts % period])

    pred1 = np.array(pred1) * scaling_factor
    pred2 = np.array(pred2) * scaling_factor
    pred3 = np.array(pred3) * scaling_factor
    pred4 = np.array(pred4) * scaling_factor
    pred5 = np.array(pred5) * scaling_factor
    # pred6 = np.array(pred6) * scaling_factor
    pred7 = np.array(pred7) * scaling_factor

    plt.plot(data, marker='o', label='data')
    plt.plot(pred3, marker='o', linewidth='8', label='partition - median', alpha=0.4)
    plt.plot(pred1, marker='o', label='partition - median - change between partitions')
    plt.plot(pred7, marker='o', label='partition - median - relative change between partitions')
    plt.plot(pred2, marker='o', label='change between timesteps - partition - median')
    plt.plot(pred4, marker='o', label='change between timesteps - partition - mean')
    plt.plot(pred5, marker='o', label='relative change between timesteps - partition - median')
    plt.plot([0, len(pred2) - 1], [pred2[0], scaling_factor * np.sum(med_absolute_change) / 12 * (len(pred2) - 1) + pred2[0]], marker='o', label=f'y = {scaling_factor * np.sum(med_absolute_change) / 12} x + {pred2[0]}')
    plt.plot(pred6, marker='o', label='change between timesteps - partition - 0 centered median')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{out_dir}{plot_no}_{modifire}.png')
    plt.close()
    plot_no += 1

    print(np.sum(med_absolute_change))
    print(pred2[0] - pred2[11])

    return plot_no


def compute_partitioned_mean_std(data, period, plot_no, modifire='Rainfall'):
    partitions = {}
    means = []
    std = []
    months = []

    for partition in range(period):
        partitions[partition] = []
        std.append(0)
        means.append(0)

    for idx, val in enumerate(data):
        partitions[idx % period].append(val)
        months.append(idx % period + 1)

    for partition, vals in partitions.items():
        means[partition] = sum(vals) / len(vals)
        std[partition] = np.std(vals)

    means = np.array(means)
    std = np.array(std)

    df = pd.DataFrame({'Month': months, modifire: data})
    sns.boxplot(data=df, x='Month', y=modifire)
    plt.title(f'Monthly {modifire} Distribution')
    plt.savefig(f'{out_dir}{plot_no}_Monthly_{modifire}_Distribution - box.png')
    plot_no += 1
    plt.close()
    sns.scatterplot(data=df, x='Month', y=modifire)
    plt.title(f'Monthly {modifire} Distribution')
    ticks = [idx for idx in range(1, period + 1)]
    plt.xticks(ticks, ticks)
    plt.tight_layout()
    plt.savefig(f'{out_dir}{plot_no}_Monthly_{modifire}_Distribution - points.png')
    plot_no += 1
    plt.close()
    sns.violinplot(data=df, x='Month', y=modifire, scale='count', bw=.15, inner='box')
    plt.title(f'Monthly {modifire} Distribution')
    plt.savefig(f'{out_dir}{plot_no}_Monthly_{modifire}_Distribution - violin.png')
    plot_no += 1
    plt.close()
    # sns.lineplot(data=df, x='Month', y=modifire, estimator='mean', marker='o')
    # plt.title(f'{modifire} {month_names[partition]}')

    # for partition, vals in partitions.items():
    #     partitions[partition] = np.array(vals)
    #     partitions[partition][partitions[partition] > means[partition] + std[partition]] = 0#means[partition] + std[partition]
    #     partitions[partition][partitions[partition] < means[partition] - std[partition]] = 0#means[partition] - std[partition]
    #     means[partition] = sum(partitions[partition]) / len(partitions[partition])
    #     std[partition] = np.std(partitions[partition])

    lr = LinearRegression()

    for partition, vals in partitions.items():
        # x = [partition + 1 + idx * 12 for idx in range(len(vals))]
        # x = [partition + 1 + idx for idx in range(len(vals))]
        x = [(1958 + idx) + (partition + 1) / 10 for idx in range(len(vals))]
        lr.fit(np.array(x).reshape(-1, 1), vals)
        plt.plot(x, vals, marker='o', label=modifire)
        plt.plot([x[0], x[-1]], [lr.predict([[x[0]]]), lr.predict([[x[-1]]])], label='Linear regression fit')
        plt.plot([x[0], x[-1]], [means[partition], means[partition]], label=f'Average {modifire}')
        plt.plot([x[0], x[-1]], [means[partition] + std[partition], means[partition] + std[partition]], label='Average + 1 std')
        plt.plot([x[0], x[-1]], [means[partition] - std[partition], means[partition] - std[partition]], label='Average - 1 std')
        plt.fill_between([x[0], x[-1]], [means[partition] + std[partition], means[partition] + std[partition]],
                         [means[partition] - std[partition], means[partition] - std[partition]], label='1 std', alpha=0.1, color='b')
        plt.xlabel('Year')
        plt.ylabel(modifire)
        plt.title(f'{modifire} {month_names[partition]}')
        plt.legend()
        # plt.show()
        plt.savefig(f'{out_dir}{plot_no}_{modifire} - {month_names[partition]}.png')
        plot_no += 1
        plt.close()
        plt.hist(vals)
        plt.xlabel(modifire)
        plt.ylabel('Number of Years')
        plt.title(f'{modifire} Distribution - {month_names[partition]}')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{out_dir}{plot_no}_{modifire}_Distribution - {month_names[partition]}.png')
        plot_no += 1
        plt.close()

    for partition in range(len(partitions)):
        x = partitions[partition]
        y = partitions[(partition + 1) % 12]
        plt.scatter(x, y)
        lr.fit(np.array(x).reshape(-1, 1), y)
        plt.plot([min(x), max(x)], [lr.predict([[min(x)]]), lr.predict([[max(x)]])], label='Linear regression fit', color='r')
        plt.plot([min(x), max(x)], [min(x), max(x)], label='y = x', color='g')
        plt.xlabel(f'{month_names[partition]} {modifire}')
        plt.ylabel(f'{month_names[(partition + 1) % 12]} {modifire}')
        plt.title(f'{modifire} {month_names[partition]} vs. {month_names[(partition + 1) % 12]}')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{out_dir}{plot_no}_{modifire}_{month_names[partition]}_vs._{month_names[(partition + 1) % 12]}.png')
        plot_no += 1
        plt.close()

    # print(partitions)
    # print(means + std)
    # print(means)
    # print(means - std)
    # print(std)
    periods = [month for month in range(1, period + 1)]
    plt.plot(periods, means, label=f'Average {modifire}', color='r', marker='o')
    plt.fill_between(periods, means + std, means - std, label='1 std', alpha=0.2, color='b')
    plt.legend()
    plt.title(f'Average Monthly {modifire}')
    plt.xlabel('Month')
    plt.ylabel(f'Average {modifire}')
    # plt.plot(std)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{out_dir}{plot_no}_average_monthly_{modifire}.png')
    plot_no += 1
    plt.close()

    return plot_no

def stacked_rainfall_plot(data, period, num_stacked_years, plot_no, modifire='Rainfall'):
    yearly_data = {}
    data_matrix = []
    years = []
    for year in range(len(data) // period):
        yearly_data[year] = data[year * period : (year + 1) * period]
        data_matrix.append(yearly_data[year])
        years.append(1958 + year)

    x = [month for month in range(1, period + 1)]
    for year in range(len(yearly_data) - num_stacked_years + 1):
        title = f'{modifire} for -'
        for stack_num in range(num_stacked_years):
            plt.plot(x, yearly_data[year + stack_num], label=1958+year+stack_num, marker='o')
            title += ' ' + str(1958+year+stack_num)
        plt.legend()
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel(f'Average {modifire}')
        plt.tight_layout()
        plt.savefig(f'{out_dir}{plot_no}_{title}.png')
        plot_no += 1
        plt.close()

    data_matrix = np.array(data_matrix).T

    fig, ax = plt.subplots()
    # im = ax.imshow(data_matrix)
    sns.heatmap(data_matrix)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel(f'{modifire}', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(years)))
    ax.set_yticks(np.arange(len(x)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(years)
    ax.set_yticklabels(x)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(years)):
    #     for j in range(len(x)):
    #         text = ax.text(j, i, rainfall_matrix[i][j],
    #                        ha="center", va="center", color="w")

    ax.set_title(f'{modifire} Year-Month')
    fig.tight_layout()
    # plt.show()
    fig.savefig(f'{out_dir}{plot_no}_{modifire}_heatmap.png')
    plot_no += 1
    plt.close()

    return plot_no


def this_month_next_month(data, plot_no, modifire='Rainfall'):
    this_month = [data[0]]
    next_month = []

    for month in range(1, len(data) - 1):
        next_month.append(data[month])
        this_month.append(data[month])

    next_month.append(data[-1])

    # print(len(this_month))
    # print(len(next_month))

    lr = LinearRegression()

    # plt.scatter(this_month, next_month)
    x = this_month
    y = next_month
    plt.scatter(x, y)
    lr.fit(np.array(x).reshape(-1, 1), y)
    plt.plot([min(x), max(x)], [lr.predict([[min(x)]]), lr.predict([[max(x)]])], label='Linear regression fit', color='r')
    plt.plot([min(x), max(x)], [min(x), max(x)], label='y = x', color='g')
    plt.xlabel(f'This Month {modifire}')
    plt.ylabel(f'Next Month {modifire}')
    plt.title(f'{modifire} This Month vs. Next Month')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{out_dir}{plot_no}_{modifire}_This_Month_vs._Next_Month.png')
    plot_no += 1
    plt.close()

    return plot_no

def compute_relative_increase(data):
    data = np.array(data)
    dy = np.diff(data)
    relative_change = []

    # print(data[150:160])
    # print(dy[150:160])
    for idx in range(len(dy) - 1):
        if data[idx] == 0:
            print(idx, 'zero division')
            relative_change.append(None)
            dy[idx] = np.nan
        else:
            dy[idx] /= data[idx]

    # print(dy[150:160])

# plt.plot(rain)
# plt.show()


plot_no = 1
plot_no = changes_analysis(rain[:48], plot_no, 'Rain models')

# exit()
plot_no = compute_partitioned_mean_std(rain, 12, plot_no, 'Rainfall')
# plot_no = 38
plot_no = stacked_rainfall_plot(rain, 12, 3, plot_no, 'Rainfall')
# plot_no = 98
plot_no = this_month_next_month(rain, plot_no, 'Rainfall')

dy = np.diff(rain)[0 : -11]

plot_no = compute_partitioned_mean_std(dy, 12, plot_no, 'Absolute Change')
plot_no = stacked_rainfall_plot(dy, 12, 3, plot_no, 'Absolute Change')
plot_no = this_month_next_month(dy, plot_no, 'Absolute Change')

rain = rain + 1
dy = np.diff(rain)[0 : -11]
relative_change = dy / (rain[0: -12])
# relative_change[relative_change == np.inf] = 1

# plot_no = 202
plot_no = compute_partitioned_mean_std(relative_change, 12, plot_no, 'Relative Change')
plot_no = stacked_rainfall_plot(relative_change, 12, 3, plot_no, 'Relative Change')
plot_no = this_month_next_month(relative_change, plot_no, 'Relative Change')


# for idx, r in enumerate(rain):
#     if r < 1:
#         print(idx, r)

# relative_change = dy / (rain[0 : -12] + 0.001)
# print(rain[0 : -12][relative_change > 5000])
# exit()
# plt.plot(dy / (rain[0 : -12] + np.mean(rain)))
# plt.show()
# plt.close()

# compute_relative_increase(rain[0 : -11])