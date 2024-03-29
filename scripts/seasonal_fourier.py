import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
import scipy.linalg as la
from scipy import stats

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 8})
sns.set_style("whitegrid")

np.set_printoptions(precision=3, linewidth=100)  #, formatter={'float': '{: 0.3f}'.format})


def linear_interpolate_mid_point(data):
    mid = (data[:-1] + data[1:]) / 2
    tot_len = len(data) + len(mid)
    data_interpolated = np.zeros(tot_len)
    data_interpolated[np.arange(0, tot_len, 2)] = data
    data_interpolated[np.arange(1, tot_len - 1, 2)] = mid

    return data_interpolated


def plot_predictions_with_data_distributions(LDS_pred, num_pred, df_binned, df_bin_centers, center_measure, components, prediction_step_length, type='violin', title='', file_name=''):
    name = 'Accent'
    cmap = get_cmap('tab10')
    colors = cmap.colors

    # fig, ax = plt.subplots(dpi=250, figsize=(12, 6.75))
    fig, ax = plt.subplots(dpi=250, figsize=(6.75, 2.5))
    ax.set_prop_cycle(color=colors)

    x_pred_LDS = [f'{x}' for x in np.arange(1, num_pred + 1)]

    # Preparing x axis to plot in between bins. Just a hack to generate some plots to paper
    df_binned['Observation Point str'] = df_binned['Observation Point str']\
        .apply(lambda bin: f'{(int(bin) - 1) / 2 + 1}' if int(bin) % 2 == 0 else f'{int((int(bin) - 1) / 2 + 1)}')

    if prediction_step_length == 1:
        if type == 'violin':
            #  24: width=1.2
            sns.violinplot(data=df_binned, x='Observation Point str', y='Observation', linewidth=1, width=0.8, color='lemonchiffon', alpha=1)
        elif type == 'box':
            sns.boxplot(data=df_binned, x='Observation Point str', y='Observation')

        # 12: s=3
        sns.swarmplot(data=df_binned, x='Observation Point str', y='Observation', color="k", alpha=0.8, s=2)

        ax.fill_between(df_bin_centers['Bin Point str'], df_bin_centers[center_measure] - df_bin_centers['Bin StD'],
                        df_bin_centers[center_measure] + df_bin_centers['Bin StD'], color='orangered', alpha=0.3, label='Standard Deviation')
        # ax.fill_between(df_bin_centers['Bin Point str'], df_bin_centers['Bin Median'] - df_bin_centers['Bin MAD'],
        #                 df_bin_centers['Bin Median'] + df_bin_centers['Bin MAD'], color='b', alpha=0.3, label='Median Abolute Error')
        # sns.lineplot(x=df_bin_centers['Bin Point str'], y=df_bin_centers['Bin Median'], color='b', linewidth=1, label='Bin Median', alpha=1, marker='o')
        sns.lineplot(x=df_bin_centers['Bin Point str'], y=df_bin_centers[center_measure], color='orangered', linewidth=1, label=center_measure, marker='D')
    else:
        print('***** WARNING: Cannot produce distribution plots for prediction step sizes other than 1 *****')
        title = '***** WARNING: prediction step sizes $\\neq$ 1 *****\nDistributions Cannot be aligned with Predictions'

    sns.lineplot(x=x_pred_LDS, y=LDS_pred[-2, :], label='LDS value', marker='o', color='r', linewidth=2)

    title += f' $(k = {components})$'
    # plt.title(title)
    # plt.xticks([f'{b}' for b in np.arange(1, 1 + len(df_bin_centers['Bin Point num'])) / 2 + 0.5])
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.tight_layout()
    plt.legend()

    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_according_to_2pi_domain(x_2pi, df_bin_centers, center_measure, continuous_pred, LDS_pred, num_pred, L, num_datapoints, step_length, title='', plot_derivatives=False, file_name=''):
    name = 'Accent'
    cmap = get_cmap('tab10')
    colors = cmap.colors

    # fig, ax = plt.subplots(dpi=250, figsize=(12, 6.75))
    fig, ax = plt.subplots(dpi=250, figsize=(6.75, 2.5))
    ax.set_prop_cycle(color=colors)

    x_pred_LDS = np.arange(0, num_pred * step_length, step_length) * 2 * L / (num_datapoints) - L
    # x_2pi += L

    x = list(df_bin_centers['Bin Point num'])
    x += [2 * x[-1] - x[-2]]
    y = list(df_bin_centers[center_measure])
    y += [y[0]]
    # y_median = list(df_bin_centers['Bin Median'])
    # y_median += [y_median[0]]

    x_LDS = list(x_pred_LDS)
    x_LDS += [2 * x_LDS[-1] - x_LDS[-2]]
    y_LDS = list(LDS_pred[-2, :])
    y_LDS += [y_LDS[0]]

    # sns.lineplot(x=df_bin_centers['Bin Point num'], y=df_bin_centers[center_measure], color='y', linewidth=3, label=center_measure, marker='D', alpha=0.8)
    sns.lineplot(x=x, y=y, color='y', linewidth=3, label=center_measure, marker='D', alpha=0.8)
    # sns.lineplot(x=x, y=y_median, color='g', linewidth=1.5, label='Bin Median', marker='*', markersize=7, alpha=0.8)
    sns.lineplot(x=x_2pi, y=continuous_pred, label='LDS ($\delta t \ll$)', linewidth=2, color='k')

    # sns.lineplot(x=x_pred_LDS, y=LDS_pred[-2, :], label=f'LDS ($\delta t = {step_length}$)', marker='o', color='r', linewidth=1)
    sns.lineplot(x=x_LDS, y=y_LDS, label=f'LDS ($\delta t = {step_length}$)', marker='o', color='r', linewidth=1)

    if plot_derivatives:
        # x_pred_LDS += (x_pred_LDS[1] - x_pred_LDS[0]) / 2
        sns.lineplot(x=x_pred_LDS[: -1], y=np.diff(LDS_pred[-2, :]), label=f'diff( LDS ($\delta t = {step_length}$) )', marker='o', color='r', alpha=0.5, linewidth=0.5)
        sns.lineplot(x=x_pred_LDS[: -1], y=LDS_pred[-1, : -1], label='LDS derivative', marker='o', color='k', alpha=0.5, linewidth=0.5)

    xticks = [f'{(bin - 1) / 2 + 1}' if bin % 2 == 0 else f'{int((bin - 1) / 2 + 1)}' for bin in list(range(1, 1 + len(df_bin_centers['Bin Point num']))) + [1]]
    # plt.xticks(df_bin_centers['Bin Point num'], range(1, 1 + len(df_bin_centers['Bin Point num'])))
    # plt.xticks(x, list(range(1, 1 + len(df_bin_centers['Bin Point num']))) + [1])
    plt.xticks(x, xticks)

    # plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.tight_layout()
    plt.legend()

    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_sinusoidals(sinus_df, period, file_name=''):
    fig, ax = plt.subplots(dpi=250, figsize=(24, 6.75))
    x_highlight = generate_x(L, period + 1)
    sns.set(rc={"lines.linewidth": 0.7})
    g = sns.FacetGrid(sinus_df, col='Type', row='Frequency', margin_titles=True)
    g.map(sns.lineplot, 'Angle', 'Value')
    axes = g.axes
    for ax_row in axes:
        for ax_col in ax_row:
            for start in np.arange(0, period, 2):
                ax_col.axvspan(x_highlight[start], x_highlight[start + 1], color="green", alpha=0.3)
                l = ax_col.get_lines()[0]  # get the relevant Line2D object
                # l.set_linewidth(0.7)
                l.set_color('r')
    g.set_titles(col_template='{col_name}', row_template='$\omega = {row_name}$')

    plt.xticks(x_highlight, range(1, 1 + len(x_highlight)))
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def sinusoidals_to_df(x, sinusoidals, components):
    sinus = []
    for comp in range(components):
        freq = comp + 1
        for i, s in enumerate(sinusoidals[2 * comp, :]):
            sinus.append({'Type': 'sin', 'Frequency': freq, 'Angle': x[i], 'Value': s})
        for i, c in enumerate(sinusoidals[2 * comp + 1, :] / freq):
            sinus.append({'Type': 'cos', 'Frequency': freq, 'Angle': x[i], 'Value': c})

    sinus_df = pd.DataFrame(sinus)
    return sinus_df


def discretize_line(y1, y2, spl):
    dx = 1 / (spl)
    x = np.arange(0, 1, dx)
    return list(y1 + (y2 - y1) * x)


def linear_interpolate_data_sequence(data, spl):
    f = []
    for idx in range(len(data)):
        f += discretize_line(data[idx], data[(idx + 1) % len(data)], spl)
    return f


def generate_x(L, num_points):
    x = np.array([2 * L / (num_points - 1) * idx - L for idx in range(num_points)])
    return x


def get_magnitudes(C, D):
    mag = np.zeros(len(C))
    vec = np.zeros(2)

    for i, Ci in enumerate(C):
        vec[0] = Ci
        vec[1] = D[i]
        mag[i] = np.sqrt(np.dot(vec, vec))

    return mag


# Full blown LDS vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def assemble_sinusoidal_generating_full_blown_LDS(components):
    comp4 = components * 4
    A = np.zeros((comp4, comp4))
    s0 = np.zeros((comp4, 1))

    for i in range(components):
        ip1 = i + 1
        i4 = i * 4
        i4p1 = i4 + 1
        i4p2 = i4 + 2
        i4p3 = i4 + 3

        A[i4, i4p1] = 1
        A[i4p2, i4p3] = 1
        A[i4p1, i4] = -ip1**2
        A[i4p3, i4p2] = -ip1**2

        s0[i4p1][0] = ip1 * np.cos(-ip1 * np.pi)  # (-1)**i*ip1
        s0[i4p2][0] = np.cos(-ip1 * np.pi)  # (-1)**(ip1)

    return A, s0


def assemble_complete_full_blown_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
    '''
        dx * L * spl = 2 * pi / (m - 1)
    '''
    dim = 4 + A_sinusoidal.shape[0]
    A_base = np.zeros((dim, dim))
    # A[0][0] = 1
    A_base[2:2 + A_sinusoidal.shape[0], 2:2 + A_sinusoidal.shape[0]] = A_sinusoidal
    A_base[2 + A_sinusoidal.shape[0]][0] = 0
    A_base[np.ix_([-2], np.arange(3, dim - 4, 4))] = D  # -2 = 2 + A_sinusoidal.shape[0]
    A_base[np.ix_([-2], np.arange(5, dim - 2, 4))] = C

    for component in np.arange(components):
        components4 = component * 4
        componentsp1 = component + 1
        A_base[-1][components4 + 2] = -(componentsp1 ** 2) * D[component]
        A_base[-1][components4 + 4] = -(componentsp1 ** 2) * C[component]

    s0 = np.zeros((dim, 1))
    s0[0][0] = 1  # C0 / 2
    s0[2:2 + A_sinusoidal.shape[0], 0] = s0_sinusoidal.T

    ft_coef = np.zeros(4 + A_sinusoidal.shape[0])
    ft_coef[0] = C0 / 2  # 1#
    ft_coef[np.arange(2, dim - 5, 4)] = D
    ft_coef[np.arange(4, dim - 3, 4)] = C

    s0[-2][0] = np.matmul(ft_coef, s0)   # f(t0)
    s0[-1][0] = np.matmul(A_base[-2, :], s0)  # \dot f(t0)

    A = la.expm(A_base * dx * L * spl)

    return A_base, A, s0


def generate_sinusoidal_curves_from_full_blown_LDS(A, s0, x):
    points = np.zeros((len(s0), len(x)))
    for idx, t in enumerate(x - x[0]):
        points[:, idx] = np.matmul(la.expm(A * t), s0)[:, 0]

    return points


'''
components = 2
dx = 0.1
x = np.arange(-np.pi, np.pi + dx, dx)
A, s0 = generate_sinusoidal_generating_full_blown_LDS(components)
print(A)
print(s0)
sinusoidals = generate_sinusoidal_curves_from_full_blown_LDS(A, s0, x)
for k in range(components):
    sns.lineplot(x=x, y=sinusoidals[4 * k, :], label=f'$sin({k + 1}t)$')#, marker='o')#
    # sns.lineplot(x=x, y=sinusoidals[4 * k + 2, :], label=f'$cos({k + 1}t)$')#, marker='o'
    # sns.lineplot(x=x, y=trig_sinus[2 * k, :], label=f'$trigsin({k + 1}t)$')#, marker='o'
plt.show()
exit()
'''
# Full blown LDS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def assemble_sinusoidal_generating_compact_LDS(components, start_angle):
    comp2 = components * 2
    A = np.zeros((comp2, comp2))
    s0 = np.zeros((comp2, 1))
    for i in range(components):
        i2 = i * 2
        i2p1 = i2 + 1
        ip1 = i + 1
        A[i2  ][i2p1] = 1
        A[i2p1][i2  ] = -((ip1) ** 2)
        s0[i2p1][0] = ip1 * np.cos(ip1 * np.pi)  #(-1)**(ip1)*ip1#-ip1

    return A, s0


def generate_sinusoidal_curves(A, s0, dx, num_points, L):
    A = la.expm(A * dx * L)
    sin_t = np.zeros((len(s0), num_points))
    sin_t[:, 0] = s0[:, 0]

    for t in range(1, num_points):
        sin_t[:, t] = np.matmul(A, sin_t[:, t - 1])
        # sin_t[:, t] = np.matmul(la.expm(A * t * dx * L), s0)[:, 0]

    return A, sin_t


def generate_sinusoidal_curves_from_trig_functions(x, components, num_points, L):
    sin_t = np.zeros((2 * components, num_points))

    for k in range(components):
        sin_t[2 * k + 1, :] = np.cos(np.pi * (k+1) * x / L)
        sin_t[2 * k, :] = np.sin(np.pi * (k+1) * x / L)

    return sin_t


def compute_fourier_coefficients_from_trig_functions(x, f, dx, components, L):
    C = np.zeros(components)
    D = np.zeros(components)

    C0 = np.sum(f * np.ones_like(x)) * dx

    for k in range(components):
        cos_nx = np.cos(np.pi * (k+1) * x / L)
        sin_nx = np.sin(np.pi * (k+1) * x / L)

        C[k] = np.sum(f * cos_nx) * dx  # Inner product
        D[k] = np.sum(f * sin_nx) * dx

    return C0, C, D


def compute_fourier_coefficients_from_LDS_sinusoidals(x, f, dx, components, sinusoidals, full_blown=False):
    C = np.zeros(components)
    D = np.zeros(components)

    C0 = np.sum(f * np.ones_like(x)) * dx

    if full_blown:
        print('Computing Fourier coefficients using the FULL BLOWN LDS')
        multiplier = 4
        offset = 2
    else:
        print('Computing Fourier coefficients using the COMPACT LDS')
        multiplier = 2
        offset = 1

    # f0 = C0/2
    # f0_dot = 0

    for k in range(components):
        cos_nx = sinusoidals[multiplier * k + offset, :]
        sin_nx = sinusoidals[multiplier * k, :]

        if full_blown:
            C[k] = np.sum(f * cos_nx) * dx  # Inner product
        else:
            C[k] = np.sum(f * cos_nx) * dx / (k+1)  # **2  # Inner product

        D[k] = np.sum(f * sin_nx) * dx

        # f0 += C[k] * cos_nx[0] + D[k] * sin_nx[0]
        # f0 += C[k] * sinusoidals[4 * k + 2, :][0] + D[k] * sinusoidals[4 * k, :][0]
        # f0_dot += C[k] * sinusoidals[4 * k + 3, :][0] + D[k] * sinusoidals[4 * k + 1, :][0]

    return C0, C, D


def assemble_complete_compact_LDS(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
    '''
        dx * L * spl = 2 * pi / (m - 1)
    '''
    A_sinusoidal = la.expm(A_sinusoidal * dx * L * spl)
    dim = 3 + A_sinusoidal.shape[0]
    A = np.zeros((dim, dim))
    A[0][0] = 1
    A[1:1 + A_sinusoidal.shape[0], 1:1 + A_sinusoidal.shape[0]] = A_sinusoidal
    A[1 + A_sinusoidal.shape[0]][0] = C0 / 2
    A[np.ix_([-2], np.arange(1, dim - 2, 2))] = D  # -2 = 1 + A_sinusoidal.shape[0]
    A[np.ix_([-1], np.arange(2, dim - 1, 2))] = D  # -1 = 2 + A_sinusoidal.shape[0]

    for i, Ci in enumerate(C):
        i2 = i * 2
        ip1 = i + 1
        A[-2][i2 + 2] = Ci / (i + 1)
        A[-1][i2 + 1] = -ip1 * Ci

    s0 = np.zeros((dim, 1))
    s0[0][0] = 1
    s0[1:1 + A_sinusoidal.shape[0], 0] = s0_sinusoidal.T
    s0 = np.matmul(A, s0)

    return A, s0


def assemble_complete_compact_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
    dim = 4 + A_sinusoidal.shape[0]
    A_base = np.zeros((dim, dim))
    A_base[2:2 + A_sinusoidal.shape[0], 2:2 + A_sinusoidal.shape[0]] = A_sinusoidal

    A_base[np.ix_([-2], np.arange(3, dim - 1, 2))] = D  # -2 = 2 + A_sinusoidal.shape[0]

    ft_coef = np.zeros(dim)
    ft_coef[0] = C0 / 2  # 1#
    ft_coef[np.arange(2, dim - 2, 2)] = D

    for i, Ci in enumerate(C):
        i2 = i * 2
        ip1 = i + 1
        A_base[-2][i2 + 2] = -ip1 * Ci

        A_base[-1][i2 + 2] = -(ip1 ** 2) * D[i]
        A_base[-1][i2 + 3] = -ip1 * Ci

        ft_coef[i2 + 3] = Ci / (i + 1)

    s0 = np.zeros((dim, 1))
    s0[0][0] = 1#C0 / 2
    s0[2:2 + A_sinusoidal.shape[0], 0] = s0_sinusoidal.T

    s0[-2][0] = np.matmul(ft_coef, s0)   # f(t0)
    s0[-1][0] = np.matmul(A_base[-2, :], s0)  # \dot f(t0)

    A = la.expm(A_base * dx * L * spl)

    return A_base, A, s0


def fourier_curve_from_trig_functions(C0, C, D, x, L):
    fFS = C0 / 2

    for k in range(components):
        cos_nx = np.cos(np.pi * (k+1) * x / L)
        sin_nx = np.sin(np.pi * (k+1) * x / L)

        fFS = fFS + C[k] * cos_nx + D[k] * sin_nx

        # ax.plot(x, sin_nx, '-')
        # ax.plot(x, cos_nx, '-')

    return fFS


def fourier_curve_from_LDS(A, s0, num_pred):
    curves = np.zeros((len(s0), num_pred))
    curves[:, 0] = s0[:, 0]
    for t in range(1, num_pred):
        curves[:, t] = np.matmul(A, curves[:, t - 1])

    return curves[np.r_[-2:0], :]


def fourier_curve_from_LDS_with_more_points(A, s0, x):
    curves = np.zeros((len(s0), len(x)))
    for idx, t in enumerate(x - x[0]):
        curves[:, idx] = np.matmul(la.expm(A * t), s0)[:, 0]

    return curves[-2, :]


# vvvvvvvvvvvvvvvvvvvv Estimating Fourier coefficients using least squares vvvvvvvvvvvvvvvvvvvvv

# Partitioning data into bins within a period
def partition_data_according_to_period(data, timesteps, period=12, L=np.pi):
    partitions = {}
    means = []
    stds = np.zeros(period)
    medians = []
    mads = np.zeros(period)
    observation_point_with_in_a_period_str = []
    observation_point_with_in_a_period_num = []
    bin_location_with_in_a_period_str = []
    bin_location_with_in_a_period_num = []
    observation_point_with_in_2pi = generate_x(L, period + 1)  # + L

    for partition in range(period):
        partitions[partition] = []
        means.append(0)
        medians.append(0)

    for idx, val in enumerate(data):
        observation_point = timesteps[idx] % period
        partitions[observation_point].append(val)
        observation_point_with_in_a_period_num.append(observation_point_with_in_2pi[observation_point])
        observation_point_with_in_a_period_str.append(f'{observation_point + 1}')

    for partition, vals in partitions.items():
        means[partition] = np.mean(vals)
        stds[partition] = np.std(vals)
        medians[partition] = np.median(vals)
        mads[partition] = stats.median_abs_deviation(vals)
        bin_location_with_in_a_period_num.append(observation_point_with_in_2pi[partition])
        bin_location_with_in_a_period_str.append(f'{partition + 1}')

    df_binned = pd.DataFrame({'Observation Point str': observation_point_with_in_a_period_str,
                              'Observation Point num': observation_point_with_in_a_period_num,
                              'Observation': data})
    df_bin_centers = pd.DataFrame({'Bin Point str': bin_location_with_in_a_period_str,
                                   'Bin Point num': bin_location_with_in_a_period_num,
                                   'Bin Mean': means, 'Bin Median': medians,
                                   'Bin StD': stds, 'Bin MAD': mads})

    return df_bin_centers, partitions, df_binned


def compute_fourier_coefficients_from_least_square_optimization(binned_data, num_data, components, L):
    components2p1 = 2 * components + 1
    num_bins = len(binned_data) + 1
    tot_data_points = num_data + len(binned_data[0]) # Total number of data points in all the bins + the number of data points in bin 0

    x = generate_x(L, num_bins)

    # print(num_bins)
    # print(x)

    trig_sinus = generate_sinusoidal_curves_from_trig_functions(x, components, num_bins, L)
    # print(trig_sinus)

    U = np.zeros((tot_data_points, components2p1))
    y = np.zeros((tot_data_points, 1))

    U[:, 0] = 0.5

    row = 0
    for idx in range(num_bins):
        bin = idx % (num_bins - 1)
        # print(bin)

        for data_point in binned_data[bin]:
            y[row, 0] = data_point
            U[np.ix_([row], np.arange(1, components2p1))] = trig_sinus[:, bin].T
            row += 1

    # print(U)
    # print(y)

    x, residuals, rank, s = np.linalg.lstsq(U, y, rcond=None)
    # print(x)
    # print(residuals)
    # print(rank)
    # print(s)

    C0 = x[0, 0]
    C = x.T[np.ix_([0], np.arange(2, components2p1, 2))][0]
    D = x.T[np.ix_([0], np.arange(1, components2p1 - 1, 2))][0]

    return C0, C, D

# ^^^^^^^^^^^^^^^^^^^ Estimating Fourier coefficients using least squares ^^^^^^^^^^^^^^^^^^^^^

# vvvvvvvvvvvvvvvvvvvv Finding the best k vvvvvvvvvvvvvvvvvvvvv

def train_validate_test_split(data, timesteps, period):
    train = 32 * period
    validate = train + 20 * period
    train_data = data[:train]
    train_timesteps = timesteps[:train]
    validate_data = data[train: validate]
    validate_timesteps = timesteps[train: validate]
    test_data = data[validate:]
    test_timesteps = timesteps[validate:]

    return train_data, train_timesteps, validate_data, validate_timesteps, test_data, test_timesteps


def compute_rmse(pred, binned_test_data, prediction_locations=2):
    """
    :param pred:
    :param binned_test_data:
    :param prediction_locations: When 1 predictions are only at bin locations. When 2 there are predictions
                                 in between bins in addition
    :return:
    """
    binned_errors = {}
    bin_total_squared_errors = np.zeros(int(len(binned_test_data) / prediction_locations))
    bin_between_total_squared_errors = np.zeros(int(len(binned_test_data) / 2))
    bin_wise_rmse = np.zeros(len(binned_test_data))
    bin_tot_test_data = 0
    bin_between_tot_test_data = 0

    for bin, vals in binned_test_data.items():
        binned_errors[bin] = np.array(vals) - pred[bin]

    for bin, errors in binned_errors.items():
        if bin % prediction_locations == 0:
            bin_total_squared_errors[int(bin / prediction_locations)] = np.dot(errors, errors)
            bin_tot_test_data += len(errors)
            bin_wise_rmse[bin] = np.sqrt(bin_total_squared_errors[int(bin / prediction_locations)] / len(errors))
        else:
            bin_between_total_squared_errors[int((bin - 1) / 2)] = np.dot(errors, errors)
            bin_between_tot_test_data += len(errors)
            bin_wise_rmse[bin] = np.sqrt(bin_between_total_squared_errors[int((bin - 1) / 2)] / len(errors))

    bin_total_se = np.sum(bin_total_squared_errors)
    bin_between_total_se = np.sum(bin_between_total_squared_errors)
    bin_rmse = np.sqrt(bin_total_se / bin_tot_test_data)
    bin_between_rmse = np.sqrt(bin_between_total_se / bin_between_tot_test_data)
    combined_rmse = np.sqrt((bin_total_se + bin_between_total_se) / (bin_tot_test_data + bin_between_tot_test_data))

    return bin_rmse, bin_between_rmse, combined_rmse, bin_wise_rmse


def find_best_k(data, timesteps, period, L):
    file_name = ''
    center_measure = 'Bin Mean'
    spl = 1
    dx = 2 / period

    num_full_spls_to_predict = period
    prediction_step_length = 0.5
    num_points_to_predict = int(np.ceil(num_full_spls_to_predict / prediction_step_length))

    prediction_locations = 1 / prediction_step_length
    period_validation = period * int(prediction_locations)

    train_data, train_timesteps, validate_data, validate_timesteps, test_data, test_timesteps = train_validate_test_split(data, timesteps, period)
    # train_data = data
    # train_timesteps = timesteps

    num_data = len(train_data)

    train_df_bin_centers, train_binned_data, train_df_binned = partition_data_according_to_period(train_data, train_timesteps, period, L)

    m = len(train_df_bin_centers[center_measure])  # => number of line segments = m-1 = 4
    f = linear_interpolate_data_sequence(train_df_bin_centers[center_measure], spl=10)
    x = generate_x(L, len(f))

    if prediction_locations == 2:
        validate_data = linear_interpolate_mid_point(validate_data)
        validate_timesteps = np.arange(len(validate_timesteps) * 2 - 1)

        # test_data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12])
        # means = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        # test_timesteps = np.array(range(24))
        test_data = linear_interpolate_mid_point(test_data)
        test_timesteps = np.arange(len(test_timesteps) * 2 - 1)

        naive_mean_predictions = np.zeros(len(train_df_bin_centers['Bin Mean']) + 1)
        naive_median_predictions = np.zeros(len(train_df_bin_centers['Bin Median']) + 1)

        naive_mean_predictions[:-1] = np.array(train_df_bin_centers['Bin Mean'])
        naive_median_predictions[:-1] = np.array(train_df_bin_centers['Bin Median'])

        naive_mean_predictions[-1] = train_df_bin_centers['Bin Mean'][0]
        naive_median_predictions[-1] = train_df_bin_centers['Bin Median'][0]

        naive_mean_predictions = linear_interpolate_mid_point(naive_mean_predictions)[:-1]
        naive_median_predictions = linear_interpolate_mid_point(naive_median_predictions)
    else:
        naive_mean_predictions = train_df_bin_centers['Bin Mean']
        naive_median_predictions = train_df_bin_centers['Bin Median']

    validate_df_bin_centers, validate_binned_data, validate_df_binned = partition_data_according_to_period(validate_data, validate_timesteps, period_validation, L)
    test_df_bin_centers, test_binned_data, test_df_binned = partition_data_according_to_period(test_data, test_timesteps, period_validation, L)

    naive_mean_bin_rmses, naive_mean_bin_between_rmses, naive_mean_combined_rmse, _ = compute_rmse(naive_mean_predictions, test_binned_data, prediction_locations=prediction_locations)
    naive_median_bin_rmses, naive_median_bin_between_rmses, naive_median_combined_rmse, _ = compute_rmse(naive_median_predictions, test_binned_data, prediction_locations=prediction_locations)
    print(f'RMSE Naive Bin Means  : \n\tBin     : {naive_mean_bin_rmses:.2f}\n\tBetween : {naive_mean_bin_between_rmses:.2f}\n\tCombined: {naive_mean_combined_rmse:.2f}')
    print(f'RMSE Naive Bin Medians  : \n\tBin     : {naive_median_bin_rmses:.2f}\n\tBetween : {naive_median_bin_between_rmses:.2f}\n\tCombined: {naive_median_combined_rmse:.2f}')
    # return

    highest_frequency = int(period / 2)
    bin_rmses = np.zeros(highest_frequency)
    bin_between_rmses = np.zeros(highest_frequency)
    combined_rmses = np.zeros(highest_frequency)
    bin_wise_rmses = []

    test_bin_rmses = np.zeros(highest_frequency)
    test_bin_between_rmses = np.zeros(highest_frequency)
    test_combined_rmses = np.zeros(highest_frequency)

    for components in range(1, highest_frequency + 1):
        print(components)
        A_sinusoidal, s0_sinusoidal = assemble_sinusoidal_generating_compact_LDS(components, train_data[0])
        C0, C, D = compute_fourier_coefficients_from_least_square_optimization(binned_data=train_binned_data,
                                                                               num_data=num_data,
                                                                               components=components,
                                                                               L=L)
        A_base, A, s0 = assemble_complete_compact_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx=dx, L=L,
                                                                  spl=spl * prediction_step_length)
        LDS_pred = fourier_curve_from_LDS(A, s0, num_points_to_predict)
        LDS_continuous_pred = fourier_curve_from_LDS_with_more_points(A_base, s0, x)

        bin_rmses[components - 1], bin_between_rmses[components - 1], combined_rmses[components - 1], bin_wise_rmses_for_components = compute_rmse(LDS_pred[0], validate_binned_data, prediction_locations=prediction_locations)
        test_bin_rmses[components - 1], test_bin_between_rmses[components - 1], test_combined_rmses[components - 1], test_bin_wise_rmses_for_components = compute_rmse(LDS_pred[0], test_binned_data, prediction_locations=prediction_locations)

        for b in range(period_validation):
            bin_wise_rmses.append({'Components': components,
                                   'Bin': b,
                                   'Error': bin_wise_rmses_for_components[b]})

        # plot_according_to_2pi_domain(x, test_df_bin_centers, center_measure, LDS_continuous_pred, LDS_pred,
        #                              num_points_to_predict, L, m, prediction_step_length, 'Test Title',
        #                              plot_derivatives=False, file_name=f'{components}_2pi.pdf')
        # plt.close()
        # # plot_predictions_with_data_distributions(LDS_pred, num_points_to_predict, validate_df_binned, validate_df_bin_centers,
        # #                                          center_measure, components, prediction_step_length=1, type='violin',
        # #                                          title='Predictions with Data Distributions', file_name=f'{components}_dist.pdf')
        # plot_predictions_with_data_distributions(LDS_pred, num_points_to_predict, test_df_binned, train_df_bin_centers,
        #                                          center_measure, components, prediction_step_length=1, type='violin',
        #                                          title='Predictions with Data Distributions', file_name=f'{components}_dist.pdf')
        # plt.close()

    # return
    print('\n-----------------------')
    print(f'k\tBin\t\tBetween Bin\tCombined')
    print('-----------------------')
    for components in range(1, highest_frequency + 1):
        print(f'{components}\t{test_bin_rmses[components - 1]:.2f}\t{test_bin_between_rmses[components - 1]:.2f}\t{test_combined_rmses[components - 1]:.2f}')
    print('-----------------------\n')

    fig, ax = plt.subplots(dpi=250, figsize=(3.25, 2))
    # fig, ax = plt.subplots(dpi=250, figsize=(20, 15))
    zoom = 3
    sns.lineplot(x=range(zoom, highest_frequency + 1), y=bin_rmses[zoom - 1:], marker='o', markersize=5, label='RMSE for bins')
    if prediction_locations == 2:
        sns.lineplot(x=range(zoom, highest_frequency + 1), y=bin_between_rmses[zoom - 1:], marker='o', markersize=5, label='RMSE for between bins', color='r')
        sns.lineplot(x=range(zoom, highest_frequency + 1), y=combined_rmses[zoom - 1:], label='RMSE combined', color='g')

    plt.xticks(range(zoom, highest_frequency + 1))
    # plt.title('Validation Set RMSE')
    plt.xlabel('Highest Frequency ($k$)')
    plt.ylabel('RMSE')
    plt.tight_layout()
    # plt.show()
    plt.savefig('rmse_all.pdf')
    plt.close()

    df_bin_wise_rmses = pd.DataFrame(bin_wise_rmses)
    if prediction_locations == 2:
        df_bin_wise_rmses['Bin'] = df_bin_wise_rmses['Bin'].apply(lambda b: b / 2)

    # fig, ax = plt.subplots(dpi=250, figsize=(12, 6.75))
    # sns.lineplot(data=df_bin_wise_rmses, x='Components', y='Error', hue='Bin')
    # g = sns.FacetGrid(df_bin_wise_rmses, col='Bin', margin_titles=False, sharey=False, col_wrap=6)
    # g.map(sns.lineplot, 'Components', 'Error')

    # plt.show()
    # plt.close()

# ^^^^^^^^^^^^^^^^^^^ Finding the best k ^^^^^^^^^^^^^^^^^^^^^


# Define domain
# samples per line segment
spl = 1000

# Periodic data sequence
df_rain = pd.read_csv('../data/mini_use_case/TerraClimateOromiaMontlhyPrecip.csv')
rain = np.array(df_rain['(mm) Precipitation (TerraClimate) at State, 1958-01-01 to 2019-12-31'].tolist())
months = np.array(range(len(rain)))

# data = [0, 0, 1, 0, 0]
# data = [100, 100, 200, 150, 140]
# timesteps = [0, 1, 2, 3, 4]
# period = 5

data = rain
timesteps = months
period = 12

L = np.pi

num_data = len(data)

train_data, train_timesteps, validate_data, validate_timesteps, test_data, test_timesteps = train_validate_test_split(data, timesteps, period)
df_bin_centers, binned_data, df_binned = partition_data_according_to_period(train_data, train_timesteps, period, L)

find_best_k(data, timesteps, period, L)
exit()

center_measure = 'Bin Mean'  # 'Bin Median'  #

# Number of data points
m = len(df_bin_centers[center_measure])  # => number of line segments = m-1 = 4

f = linear_interpolate_data_sequence(df_bin_centers[center_measure], spl)
'''
# Experiment: Use spl based interpolated data in the least square estimation of Fourier coefficients
# Preparing data
bin_means, binned_data = partition_data_according_to_period(f, list(range(len(f))), len(f))
num_data = len(f)
period = len(f)
'''
# Total number of samples
tns = len(f)  # spl * (m - 1)

dx = 2 / (tns - 1)
x = generate_x(L, len(f))  # L * np.arange(-1, 1, dx)
dxs = np.diff(x)

# Compute Fourier series
C0 = np.sum(f * np.ones_like(x)) * dx
fFS = C0 / 2

components = 4
file_name = f'dist_{components}.pdf'
# for components in range(1, 13):
#     file_name = f'{2 * (components - 0) + 0}_k-{components}'

least_sq = True
full_blown = False

if full_blown:
    A_sinusoidal, s0_sinusoidal = assemble_sinusoidal_generating_full_blown_LDS(components)
else:
    A_sinusoidal, s0_sinusoidal = assemble_sinusoidal_generating_compact_LDS(components, x[0])

A_fun, sinusoidals = generate_sinusoidal_curves(A_sinusoidal, s0_sinusoidal, dx, len(f), L)
'''
trig_sinus = generate_sinusoidal_curves_from_trig_functions(x, components, len(f), L)
for k in range(components):
    if full_blown:
        # sns.lineplot(x=x, y=sinusoidals[4 * k, :], label=f'$sin({k + 1}t)$')#, marker='o')#
        sns.lineplot(x=x, y=sinusoidals[4 * k + 2, :], label=f'$cos({k + 1}t)$')#, marker='o'
    else:
        # sns.lineplot(x=x, y=sinusoidals[2 * k, :], label=f'$sin({k + 1}t)$')#, marker='o')#
        sns.lineplot(x=x, y=sinusoidals[2 * k + 1, :] / (k+1), label=f'$cos({k + 1}t)$')#, marker='o'
    # sns.lineplot(x=x, y=trig_sinus[2 * k, :], label=f'$trigsin({k + 1}t)$')#, marker='o'
plt.show()
exit()
'''

C0_trig, C_trig, D_trig = compute_fourier_coefficients_from_trig_functions(x, f, dx, components, L)
if least_sq:
    C0, C, D = compute_fourier_coefficients_from_least_square_optimization(binned_data=binned_data,
                                                                           num_data=num_data,
                                                                           components=components,
                                                                           L=L)
else:
    # When full_blown = True, this uses the full blown LDS version. Otherwise the compact LDS version
    C0, C, D = compute_fourier_coefficients_from_LDS_sinusoidals(x, f, dx, components, sinusoidals, full_blown)

magnitudes = get_magnitudes(C, D)
print(magnitudes)
# print(C0)
# print(C)
# print(D)

# How many spl's to advance the LDS
num_full_spls_to_predict = 13
prediction_step_length = 1
num_points_to_predict = int(np.ceil(num_full_spls_to_predict / prediction_step_length))

if full_blown:
    A_base, A, s0 = assemble_complete_full_blown_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl * prediction_step_length)
    title = 'Full Blown LDS Predictions'
else:
    # A, s0 = assemble_complete_compact_LDS(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl * prediction_step_length)
    A_base, A, s0 = assemble_complete_compact_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl * prediction_step_length)
    title = 'Compact LDS Predictions'

plot_derivatives = True

if plot_derivatives:
    title += ' and Derivatives'

if prediction_step_length < 1 and least_sq:
    title += '\nIntermediate Points & Least SQ'
elif prediction_step_length < 1:
    title += '\nIntermediate Points'
elif least_sq:
    title += '\nLeast SQ'
title += f' $(k = {components})$'

LDS_pred = fourier_curve_from_LDS(A, s0, num_points_to_predict)
LDS_continuous_pred = fourier_curve_from_LDS_with_more_points(A_base, s0, x)

# trig_pred = fourier_curve_from_trig_functions(C0_trig, C_trig, D_trig, x, L)

plot_according_to_2pi_domain(x, df_bin_centers, center_measure, LDS_continuous_pred, LDS_pred, num_points_to_predict, L, m, prediction_step_length, title, plot_derivatives, file_name=file_name)
# plot_predictions_with_data_distributions(LDS_pred, num_points_to_predict, df_binned, df_bin_centers, center_measure, components, prediction_step_length,
#                                          type='violin', title='Predictions with Data Distributions', file_name=file_name)
# sinus_df = sinusoidals_to_df(x, sinusoidals, components)
# plot_sinusoidals(sinus_df, period, file_name='sine_pallet.pdf')
