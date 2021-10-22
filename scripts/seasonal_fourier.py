import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import scipy.linalg as la

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})
sns.set_style("whitegrid")

np.set_printoptions(precision=3, linewidth=100)  #, formatter={'float': '{: 0.3f}'.format})

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
def generate_sinusoidal_generating_full_blown_LDS(components):
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


def generate_fourier_coefficients_from_full_blown_LDS_sinusoidals(x, f, dx, components, sinusoidals):
    C = np.zeros(components)
    D = np.zeros(components)

    C0 = np.sum(f * np.ones_like(x)) * dx

    # f0 = C0/2
    # f0_dot = 0

    for k in range(components):
        cos_nx = sinusoidals[4 * k + 2, :]
        sin_nx = sinusoidals[4 * k, :]

        C[k] = np.sum(f * cos_nx) * dx  # Inner product
        D[k] = np.sum(f * sin_nx) * dx

        # f0 += C[k] * cos_nx[0] + D[k] * sin_nx[0]
        # f0 += C[k] * sinusoidals[4 * k + 2, :][0] + D[k] * sinusoidals[4 * k, :][0]
        # f0_dot += C[k] * sinusoidals[4 * k + 3, :][0] + D[k] * sinusoidals[4 * k + 1, :][0]

    return C0, C, D


def generate_full_full_blown_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
    '''
        dx * L * spl = 2 * pi / (m - 1)
    '''
    dim = 4 + A_sinusoidal.shape[0]
    A = np.zeros((dim, dim))
    # A[0][0] = 1
    A[2:2 + A_sinusoidal.shape[0], 2:2 + A_sinusoidal.shape[0]] = A_sinusoidal
    A[2 + A_sinusoidal.shape[0]][0] = 0
    A[np.ix_([-2], np.arange(3, dim - 4, 4))] = D  # -2 = 2 + A_sinusoidal.shape[0]
    A[np.ix_([-2], np.arange(5, dim - 2, 4))] = C

    for component in np.arange(components):
        components4 = component * 4
        componentsp1 = component + 1
        A[-1][components4 + 2] = -(componentsp1 ** 2) * D[component]
        A[-1][components4 + 4] = -(componentsp1 ** 2) * C[component]
    # print(A)

    s0 = np.zeros((dim, 1))
    s0[0][0] = 1  # C0 / 2
    s0[2:2 + A_sinusoidal.shape[0], 0] = s0_sinusoidal.T

    ft_coef = np.zeros(4 + A_sinusoidal.shape[0])
    ft_coef[0] = C0 / 2  # 1#
    ft_coef[np.arange(2, dim - 5, 4)] = D
    ft_coef[np.arange(4, dim - 3, 4)] = C

    s0[-2][0] = np.matmul(ft_coef, s0)   # f(t0)
    s0[-1][0] = np.matmul(A[-2, :], s0)  # \dot f(t0)

    A = la.expm(A * dx * L * spl)

    return A, s0


def fourier_curve_from_full_blown_LDS(A, s0, num_pred, L, num_datapoints, step_length):
    curves = np.zeros((len(s0), num_pred))
    curves[:, 0] = s0[:, 0]
    for t in range(1, num_pred):
        curves[:, t] = np.matmul(A, curves[:, t - 1])

    x = np.arange(0, num_pred * step_length, step_length) * 2 * L / (num_datapoints) - L

    sns.lineplot(x=x, y=curves[-2, :], label='LDS value', marker='o', color='r', linewidth=2)
    sns.lineplot(x=x, y=curves[-1, :], label='LDS derivative', marker='o', color='b', linewidth=0.5)
    sns.lineplot(x=x[: -1], y=np.diff(curves[-2, :]), label='diff', marker='o', color='g', linewidth=0.5)

    plt.legend()
    plt.show()


# Have to update this similar to generate_sinusoidal_curves() to be used in the current pipeline
def generate_sinusoidal_curves_from_full_blown_LDS(A, s0, x):
    points = np.zeros((len(s0), len(x)))
    for idx, t in enumerate(x):
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


def generate_sinusoidal_generating_LDS(components, start_angle):
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


def generate_fourier_coefficients_from_trig_functions(x, f, dx, components, L):
    C = np.zeros(components)
    D = np.zeros(components)

    C0 = np.sum(f * np.ones_like(x)) * dx

    for k in range(components):
        cos_nx = np.cos(np.pi * (k+1) * x / L)
        sin_nx = np.sin(np.pi * (k+1) * x / L)

        C[k] = np.sum(f * cos_nx) * dx  # Inner product
        D[k] = np.sum(f * sin_nx) * dx

    return C0, C, D


def generate_fourier_coefficients_from_LDS_sinusoidals(x, f, dx, components, sinusoidals):
    C = np.zeros(components)
    D = np.zeros(components)

    C0 = np.sum(f * np.ones_like(x)) * dx

    for k in range(components):
        cos_nx = sinusoidals[2 * k + 1, :]
        sin_nx = sinusoidals[2 * k, :]

        C[k] = np.sum(f * cos_nx) * dx / (k+1)#**2  # Inner product
        D[k] = np.sum(f * sin_nx) * dx

    return C0, C, D


def generate_full_LDS(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
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


def generate_full_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
    dim = 4 + A_sinusoidal.shape[0]
    A = np.zeros((dim, dim))
    A[2:2 + A_sinusoidal.shape[0], 2:2 + A_sinusoidal.shape[0]] = A_sinusoidal

    A[np.ix_([-2], np.arange(3, dim - 1, 2))] = D  # -2 = 2 + A_sinusoidal.shape[0]

    ft_coef = np.zeros(dim)
    ft_coef[0] = C0 / 2  # 1#
    ft_coef[np.arange(2, dim - 2, 2)] = D

    for i, Ci in enumerate(C):
        i2 = i * 2
        ip1 = i + 1
        A[-2][i2 + 2] = -ip1 * Ci

        A[-1][i2 + 2] = -(ip1 ** 2) * D[i]
        A[-1][i2 + 3] = -ip1 * Ci

        ft_coef[i2 + 3] = Ci / (i + 1)

    s0 = np.zeros((dim, 1))
    s0[0][0] = 1#C0 / 2
    s0[2:2 + A_sinusoidal.shape[0], 0] = s0_sinusoidal.T

    s0[-2][0] = np.matmul(ft_coef, s0)   # f(t0)
    s0[-1][0] = np.matmul(A[-2, :], s0)  # \dot f(t0)

    A = la.expm(A * dx * L * spl)

    return A, s0


def fourier_curve_from_trig_functions(C0, C, D, x, L):
    fFS = C0 / 2

    for k in range(components):
        cos_nx = np.cos(np.pi * (k+1) * x / L)
        sin_nx = np.sin(np.pi * (k+1) * x / L)

        fFS = fFS + C[k] * cos_nx + D[k] * sin_nx

        # ax.plot(x, sin_nx, '-')
        # ax.plot(x, cos_nx, '-')

    sns.lineplot(x=x, y=fFS, label='Trig function', linewidth=3, color='k')


def fourier_curve_from_LDS(A, s0, num_pred, L, num_datapoints, step_length):
    curves = np.zeros((len(s0), num_pred))
    curves[:, 0] = s0[:, 0]
    for t in range(1, num_pred):
        curves[:, t] = np.matmul(A, curves[:, t - 1])

    x = np.arange(0, num_pred * step_length, step_length) * 2 * L / (num_datapoints) - L
    sns.lineplot(x=x, y=curves[-2, :], label='LDS value', marker='o', color='r', linewidth=2)
    sns.lineplot(x=x, y=curves[-1, :], label='LDS derivative', marker='o', color='b', linewidth=0.5)
    sns.lineplot(x=x[: -1], y=np.diff(curves[-2, :]), label='diff', marker='o', color='g', linewidth=0.5)

    plt.legend()
    plt.show()



# vvvvvvvvvvvvvvvvvvvv Estimating Fourier coefficients using least squares vvvvvvvvvvvvvvvvvvvvv

# Partitioning data into bins within a period
def partition_data_according_to_period(data, timesteps, period=12):
    partitions = {}
    means = []

    for partition in range(period):
        partitions[partition] = []
        means.append(0)

    for idx, val in enumerate(data):
        partitions[timesteps[idx] % period].append(val)

    for partition, vals in partitions.items():
        means[partition] = sum(vals) / len(vals)

    return means, partitions


def compute_fourier_coefficients_from_least_square_optimization(binned_data, num_data, components, L):
    # binned_data = {0: [100], 1: [100], 2: [200], 3: [150], 4: [140]}  # [100, 100, 200, 150, 140]
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
    #print(x)
    # print(residuals)
    # print(rank)
    # print(s)
    '''
    [[276.   ]
     [  3.461]
     [ 43.597]
     [-14.213]
     [  2.798]
     [-14.213]
     [ -2.798]]
    '''

    C0 = x[0, 0]
    C = x.T[np.ix_([0], np.arange(2, components2p1, 2))][0]
    D = x.T[np.ix_([0], np.arange(1, components2p1 - 1, 2))][0]

    return C0, C, D

'''
data = [100, 100, 200, 150, 140]
timesteps = [0, 1, 2, 3, 4]
period = 5
num_data = len(data)
bin_means, binned_data = partition_data_according_to_period(data, timesteps, period)
# print(binned_data)
components = 3
L = np.pi
compute_fourier_coefficients_from_least_square_optimizatio(binned_data=binned_data,
                                                           num_data=num_data,
                                                           components=components,
                                                           L=L)
exit()
'''
# ^^^^^^^^^^^^^^^^^^^ Estimating Fourier coefficients using least squares ^^^^^^^^^^^^^^^^^^^^^


# Define domain
# samples per line segment
spl = 1000

# Periodic data sequence
# data = [0, 0, 1, 0, 0]
data = [100, 100, 200, 150, 140]
timesteps = [0, 1, 2, 3, 4]
num_data = len(data)
period = 5

bin_means, binned_data = partition_data_according_to_period(data, timesteps, period)

# Number of data points
m = len(bin_means)  # => number of line segments = m-1 = 4

f = linear_interpolate_data_sequence(bin_means, spl)

# Total number of samples
tns = len(f)  # spl * (m - 1)

dx = 2 / (tns - 1)
L = np.pi
x = generate_x(L, len(f))  # L * np.arange(-1, 1, dx)
dxs = np.diff(x)

fig, ax = plt.subplots()
sns.lineplot(x=x, y=f, color='y', linewidth=4, label='Original function')
name = 'Accent'
cmap = get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)


# Compute Fourier series
C0 = np.sum(f * np.ones_like(x)) * dx
fFS = C0 / 2

components = 5

least_sq = False
full_blown = True

if full_blown:
    A_sinusoidal, s0_sinusoidal = generate_sinusoidal_generating_full_blown_LDS(components)

    # Have to update generate_sinusoidal_curves_from_full_blown_LDS() to be used this way
    # A_fun, sinusoidals = generate_sinusoidal_curves_from_full_blown_LDS(A_sinusoidal, s0_sinusoidal, x)
else:
    A_sinusoidal, s0_sinusoidal = generate_sinusoidal_generating_LDS(components, x[0])

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

C0_trig, C_trig, D_trig = generate_fourier_coefficients_from_trig_functions(x, f, dx, components, L)
if least_sq:
    C0, C, D = compute_fourier_coefficients_from_least_square_optimization(binned_data=binned_data,
                                                                           num_data=num_data,
                                                                           components=components,
                                                                           L=L)
    print(C0)
    print(C)
    print(D)
    # exit()
else:
    if full_blown:
        C0, C, D = generate_fourier_coefficients_from_full_blown_LDS_sinusoidals(x, f, dx, components, sinusoidals)
    else:
        C0, C, D = generate_fourier_coefficients_from_LDS_sinusoidals(x, f, dx, components, sinusoidals)

magnitudes = get_magnitudes(C, D)
print(magnitudes)
# print(C0)
# print(C)
# print(D)
# C0 = 276.
# C = [43.597, 2.798, -2.798]
# D = [3.461, -14.213, -14.213]
'''
[[276.   ]
 [  3.461]
 [ 43.597]
 [-14.213]
 [  2.798]
 [-14.213]
 [ -2.798]]
'''
# exit()

# How many spl's to advance the LDS
num_full_spls_to_predict = 15
prediction_step_length = 0.5
points_to_predict = int(np.ceil(num_full_spls_to_predict / prediction_step_length))

if full_blown:
    A, s0 = generate_full_full_blown_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl * prediction_step_length)
    # A, s0 = generate_full_full_blown_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0_trig, C_trig, D_trig, dx, L, spl * prediction_step_length)
else:
    # A, s0 = generate_full_LDS(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl * prediction_step_length)
    A, s0 = generate_full_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl * prediction_step_length)

fourier_curve_from_trig_functions(C0_trig, C_trig, D_trig, x, L)
if full_blown:
    fourier_curve_from_full_blown_LDS(A, s0, points_to_predict, L, m, prediction_step_length)
else:
    fourier_curve_from_LDS(A, s0, points_to_predict, L, m, prediction_step_length)