import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})

out_dir = 'plots/rainfall/'
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

# print(len(rain))
# print(len(rain)/
# df_rain.rename(columns={'(mm) Precipitation (TerraClimate) at State, 1958-01-01 to 2019-12-31': 'Rainfall (mm)'})
# df_rain['DateTime'] = df_rain['DateTime'].apply(lambda dt : pd.to_datetime(dt))
# df_rain['Month'] = df_rain['DateTime'].apply(lambda dt : dt.month)
# df_rain['Year'] = df_rain['DateTime'].apply(lambda dt : dt.year - 100)
# df_rain.to_csv('testrain.csv')
# print(df_rain.head())
# exit()


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