import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import scipy.linalg as la

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})


def discretize_line(y1, y2, spl):
    dx = 1 / (spl)
    x = np.arange(0, 1, dx)
    return list(y1 + (y2 - y1) * x)


def linear_interpolate_data_sequence(data, spl):
    f = []
    # for idx in range(len(data) - 1):
    #     f += discretize_line(data[idx], data[idx + 1], spl)
    for idx in range(len(data)):
        f += discretize_line(data[idx], data[(idx + 1) % len(data)], spl)
    return f


def generate_x(L, num_points):
    # num_points = len(f)
    x = np.array([2 * L / (num_points - 1) * idx - L for idx in range(num_points)])
    # print(num_points, len(x))
    return x


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

    # print(A)
    # print(s0)
    return (A, s0)


def generate_sinusoidal_curves(A, s0, dx, num_points, L):
    A = la.expm(A * dx * L)
    sin_t = np.zeros((len(s0), num_points))
    sin_t[:, 0] = s0[:, 0]
    #print(sin_t)
    #print(s0)
    #return
    for t in range(1, num_points):
        sin_t[:, t] = np.matmul(A, sin_t[:, t - 1])
        #sin_t.append(s_next[0][0])
        #cos_t.append(s_next[1][0]/omega)

    #print(sin_t)
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

        C[k] = np.sum(f * cos_nx) * dx / (k+1)**2  # Inner product
        D[k] = np.sum(f * sin_nx) * dx

    return C0, C, D


def generate_full_LDS(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
    '''
        dx * L * spl = 2 * pi / (m - 1)
    '''
    A_sinusoidal = la.expm(A_sinusoidal * dx * L * spl)
    #print(A_sin.shape)
    dim = 2 + A_sinusoidal.shape[0]
    A = np.zeros((dim, dim))
    A[0][0] = 1
    A[1:1 + A_sinusoidal.shape[0], 1:1 + A_sinusoidal.shape[0]] = A_sinusoidal
    A[1 + A_sinusoidal.shape[0]][0] = C0 / 2
    A[np.ix_([1 + A_sinusoidal.shape[0]], np.arange(1, dim - 1, 2))] = D
    A[np.ix_([1 + A_sinusoidal.shape[0]], np.arange(2, dim - 1, 2))] = C
    # print(A)

    s0 = np.zeros((dim, 1))
    s0[0][0] = 1
    s0[1:1 + A_sinusoidal.shape[0], 0] = s0_sinusoidal.T
    # s0[-1][0] = np.matmul(A, s0)[-1][0]
    # s0[1:1 + A_sinusoidal.shape[0], 0] = np.matmul(A_sinusoidal, s0_sinusoidal)[:, 0] #s0_sin.T
    s0 = np.matmul(A, s0)

    #print(s0[1:1+A_sin.shape[0], 0])
    # print(s0)

    return A, s0


def generate_full_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl):
    dim = 4 + A_sinusoidal.shape[0]
    A = np.zeros((dim, dim))
    # A[0][0] = 1
    A[2:2 + A_sinusoidal.shape[0], 2:2 + A_sinusoidal.shape[0]] = A_sinusoidal
    # A[2 + A_sinusoidal.shape[0]][0] = 1
    A[np.ix_([2 + A_sinusoidal.shape[0]], np.arange(2, dim - 2, 2))] = D
    A[np.ix_([2 + A_sinusoidal.shape[0]], np.arange(3, dim - 2, 2))] = C
    # print(A)

    s0 = np.zeros((dim, 1))
    s0[0][0] = C0 / 2
    s0[2:2 + A_sinusoidal.shape[0], 0] = s0_sinusoidal.T
    # s0[-2][0] = np.matmul(A, s0)[-2][0]
    # s0[2:2 + A_sinusoidal.shape[0], 0] = np.matmul(A_sinusoidal, s0_sinusoidal)[:, 0] #s0_sin.T
    #print(s0[1:1+A_sin.shape[0], 0])
    # print(s0)

    A = la.expm(A * dx * L * spl)
    # print(A)

    return A, s0


def fourier_curve_from_trig_functions(C0, C, D, x, L):
    fFS = C0 / 2

    for k in range(components):
        cos_nx = np.cos(np.pi * (k+1) * x / L)
        sin_nx = np.sin(np.pi * (k+1) * x / L)

        fFS = fFS + C[k] * cos_nx + D[k] * sin_nx

        # ax.plot(x, sin_nx, '-')
        # ax.plot(x, cos_nx, '-')
    ax.plot(x, fFS, '-', label='Trig function', linewidth=3, color='k')

    # plt.show()


def fourier_curve_from_LDS(A, s0, num_pred, L, num_datapoints):
    curves = np.zeros((len(s0), num_pred))
    curves[:, 0] = s0[:, 0]
    for t in range(1, num_pred):
        curves[:, t] = np.matmul(A, curves[:, t - 1])

    # ax.plot([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], curves[-1, :], label='LDS', marker='o', color='r', linewidth=2)

    # x = np.arange(0, num_pred) * 2 * L / (num_datapoints - 1) - L
    x = np.arange(0, num_pred) * 2 * L / (num_datapoints) - L
    ax.plot(x, curves[-1, :], label='LDS', marker='o', color='r', linewidth=2)
    # ax.scatter(x, curves[-1, :], label='LDS', marker='o', color='r', linewidth=2)

    plt.legend()
    plt.show()


# Define domain
# samples per line segment
spl = 1000

# Periodic data sequence
data = [0, 0, 1, 0, 0]

# Number of data points
m = len(data)  # => number of line segments = m-1 = 4

#for spl in range(2, 51):
f = linear_interpolate_data_sequence(data, spl)

# Total number of samples
tns = len(f)#spl * (m - 1)

# dx = 0.001
dx = 2 / (tns - 1)
L = np.pi
x = generate_x(L, len(f))#L * np.arange(-1, 1, dx)  # Changed from np.arange(-1+dx,1+dx,dx)
dxs = np.diff(x)
# print(dx*L, x[1]-x[0], x[-1]-x[-2], min(dxs), max(dxs), max(dxs)-min(dxs))
# plt.plot(dxs)
# plt.show()
# x2 = generate_x(f)#L * np.linspace(-1, 1, len(x))
# print('\t'*(len(x2)-len(f)), spl, x[-1], x2[-1], len(f), len(x), len(x2), dx, x[1]-x[0], x2[1]-x2[0], x2[-1]-x2[-2])
#
# exit()
# n = len(x)
# m = 5
# nquart = spl  #int(np.floor(n/(m-1)))
# print(n, nquart, nquart*(m-1), 2/dx)

# Define hat function
# f = np.zeros_like(x)
#f[0:nquart] = 0.5
# f[nquart:2*nquart] = (4/n)*np.arange(1,nquart+1)
# f[2*nquart:3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)

fig, ax = plt.subplots()
ax.plot(x, f, '-', color='y', linewidth=4, label='Original function')
name = 'Accent'
cmap = get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)


# Compute Fourier series
C0 = np.sum(f * np.ones_like(x)) * dx
fFS = C0 / 2

components = 5

A_sinusoidal, s0_sinusoidal = generate_sinusoidal_generating_LDS(components, x[0])
A_fun, sinusoidals = generate_sinusoidal_curves(A_sinusoidal, s0_sinusoidal, dx, len(f), L)
'''
trig_sinus = generate_sinusoidal_curves_from_trig_functions(x, components, len(f), L)
for k in range(components):
    sns.lineplot(x=x, y=sinusoidals[2 * k, :], label=f'$sin({k + 1}t)$', marker='o')#
    # sns.lineplot(x=x, y=sinusoidals[2 * k + 1, :] / (k+1), label=f'${k + 1}cos({k + 1}t)$')#, marker='o'
    # sns.lineplot(x=x, y=trig_sinus[2 * k, :], label=f'$trigsin({k + 1}t)$')#, marker='o'
plt.show()
exit()
'''

C0_trig, C_trig, D_trig = generate_fourier_coefficients_from_trig_functions(x, f, dx, components, L)
# print(C)
# print(D)
C0, C, D = generate_fourier_coefficients_from_LDS_sinusoidals(x, f, dx, components, sinusoidals)
# print(C)
# print(D)

A, s0 = generate_full_LDS(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl)
# print(A[-1, :])
# A1, s01 = generate_full_LDS_for_mat_exp(A_sinusoidal, s0_sinusoidal, C0, C, D, dx, L, spl)
# print(A1[-2:-1, :])
# exit()
fourier_curve_from_trig_functions(C0_trig, C_trig, D_trig, x, L)
fourier_curve_from_LDS(A, s0, 15, L, m)
