import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})

out_dir = 'plots/rainfall/periods/'
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

df_rain = pd.read_csv('../data/mini_use_case/CHIRPSOromiaDailyPrecip_1981-01-01_2020-08-31.csv')
df_rain.rename(columns={'(mm) Precipitation (CHIRPS) at , 1981-01-01 to 2020-10-26': 'Rain'}, inplace=True)
df_rain['DateTime'] = pd.to_datetime(df_rain['DateTime'])
df_rain['year'] = df_rain['DateTime'].dt.year
df_rain_grp = df_rain.groupby(by='year')
print(df_rain_grp)

days_year = 365
num_years = 15
start_year = 1981
block_size = 91
block_start = 0
block_end = block_start + block_size

for block_size in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,28,30,33,36,40,45,52,60,73,91,121,182,365]: #[28, 73, 91, 182, 365]:#[1, 5, 7, 14, 28, 73, 91, 182, 365]:
    rain_blocked = []
    highlight = []
    do_highlight = True
    for year, df_year in df_rain_grp:
        if start_year <= year < start_year + num_years:
            # print(year)
            rain_year = np.array(df_year['Rain'].tolist())
            days_year = len(rain_year)
            if days_year % block_size > 0:
                rain_year = rain_year[: int(block_size * np.floor(days_year // block_size) - days_year)]
            # print(len(rain_year))
            rain_blocked_year = np.reshape(rain_year, (block_size, -1))
            rain_blocked_year = np.sum(rain_blocked_year, axis=0)
            do_highlight = not do_highlight
            if do_highlight:
                start = len(rain_blocked)
                end = start + len(rain_blocked_year)
                highlight.append((start - 0.5, end - 0.5))
            rain_blocked.extend(rain_blocked_year)

    # print(len(rain_blocked))
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(x=range(len(rain_blocked)), y=rain_blocked, marker="o")
    for start, end in highlight:
        ax.axvspan(start, end, color="green", alpha=0.3)
    plt.title(f'Rainfall\nAggregation Days: {block_size}')
    plt.ylabel(f'Rainfall for {block_size} days (mm)')
    plt.xlabel(f'Blocks of {block_size} days (Aligned to years)')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{out_dir}{block_size}.png')
    plt.close()


exit()
rain = np.array(df_rain['(mm) Precipitation (CHIRPS) at , 1981-01-01 to 2020-10-26'].tolist())[:730]
# rain = np.array(df_rain['DateTime'].tolist())[:730]
# print(rain)

# plt.plot(rain)
# plt.show()

# weekly = np.reshape(rain[:-2], (7, -1))
# weekly = np.sum(weekly, axis=0)
# print(weekly)
#
# plt.plot(weekly)
# plt.show()


# weekly = np.reshape(rain[:-2], (14, -1))
# weekly = np.sum(weekly, axis=0)
# print(weekly)
#
# plt.plot(weekly)
# plt.show()

# weekly = np.reshape(rain[:-16], (21, -1))
# weekly = np.sum(weekly, axis=0)
# print(weekly)
#
# plt.plot(weekly)
# plt.show()


weekly = np.reshape(rain[:-2], (56, -1))
weekly = np.sum(weekly, axis=0)
print(weekly)

sns.lineplot(x=range(len(weekly)), y=weekly, marker="o")
plt.show()

