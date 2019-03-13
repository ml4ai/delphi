import time
import random

import numpy as np
from numba import jit, cuda


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args)
        te = time.time()
        print(f"{method.__name__}:\t{((te - ts) * 1000):2.4f}ms")
        return result

    return timed


@timeit
def regular_matrix_comp():
    A = np.zeros((1000, 10))
    for i in range(1, 1001):
        for j in range(1, 11):
            A[i-1, j-1] = (i * j) ** 2
    B = np.zeros((10, 100))
    for i in range(10):
        for j in range(100):
            B[i, j] = random.gauss(0, 1)
    C = np.zeros((1000, 100))
    for i in range(1000):
        for j in range(100):
            C[i, j] = sum([a * b for a, b in zip(A[i, :], B[:, j])])
    D = np.zeros((1000, 100))
    for i in range(1000):
        for j in range(100):
            D[i, j] = 0 if C[i, j] < 0 else 100 * C[i, j]
    return D


@timeit
def numpy_matrix_comp():
    A = np.arange(1, 10001).reshape((1000, 10))
    B = np.random.randn(10, 100)
    C = np.matmul((A * A), B)
    D = np.where(C < 0, 0, 100*C)
    return D


@timeit
@jit
def numba_matrix_comp():
    A = np.arange(1, 10001).reshape((1000, 10))
    B = np.random.randn(10, 100)
    C = np.matmul((A * A), B)
    D = np.where(C < 0, 0, 100*C)
    return D


@timeit
def petpt_numpy_cpu(MSALB, SRAD, TMAX, TMIN, XHLAI):
    TD = 0.60*TMAX+0.40*TMIN
    ALBEDO = np.where(XHLAI <= 0, MSALB, 0.23-(0.23-MSALB)*np.exp(-0.75*XHLAI))
    SLANG = SRAD*23.923
    EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)
    EO = EEQ*1.1
    EO = np.where(TMAX > 35, EEQ*((TMAX-35.0)*0.05+1.1), EO)
    EO = np.where(TMAX < 5, EEQ*0.01*np.exp(0.18*(TMAX+20.0)), EO)
    EO = EO.reshape((len(EO), 1))
    min_bound = np.full_like(EO, 0.0001)
    Full_EO = np.concatenate((EO, min_bound), axis=1)
    EO = np.max(Full_EO, axis=1)
    return EO


@timeit
@jit
def petpt_numba_cpu(MSALB, SRAD, TMAX, TMIN, XHLAI):
    TD = 0.60*TMAX+0.40*TMIN
    ALBEDO = np.where(XHLAI <= 0, MSALB, 0.23-(0.23-MSALB)*np.exp(-0.75*XHLAI))
    SLANG = SRAD*23.923
    EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)
    EO = EEQ*1.1
    EO = np.where(TMAX > 35, EEQ*((TMAX-35.0)*0.05+1.1), EO)
    EO = np.where(TMAX < 5, EEQ*0.01*np.exp(0.18*(TMAX+20.0)), EO)
    EO = EO.reshape((len(EO), 1))
    min_bound = np.full_like(EO, 0.0001)
    Full_EO = np.concatenate((EO, min_bound), axis=1)
    EO = np.max(Full_EO, axis=1)
    return EO


@timeit
@cuda.jit
def petpt_numba_gpu(MSALB, SRAD, TMAX, TMIN, XHLAI, RES):
    TD = 0.60*TMAX+0.40*TMIN
    ALBEDO = np.where(XHLAI <= 0, MSALB, 0.23-(0.23-MSALB)*np.exp(-0.75*XHLAI))
    SLANG = SRAD*23.923
    EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)
    EO = EEQ*1.1
    EO = np.where(TMAX > 35, EEQ*((TMAX-35.0)*0.05+1.1), EO)
    EO = np.where(TMAX < 5, EEQ*0.01*np.exp(0.18*(TMAX+20.0)), EO)
    EO = EO.reshape((len(EO), 1))
    min_bound = np.full_like(EO, 0.0001)
    Full_EO = np.concatenate((EO, min_bound), axis=1)
    EO = np.max(Full_EO, axis=1)
    RES = EO


MSALB = np.random.rand(100)
SRAD = np.random.randint(1, 100, size=100)
TMAX = np.random.randint(30, 40, size=100)
TMIN = np.random.randint(10, 15, size=100)
XHLAI = np.random.rand(100)
petpt_numpy_cpu(MSALB, SRAD, TMAX, TMIN, XHLAI)
petpt_numba_cpu(MSALB, SRAD, TMAX, TMIN, XHLAI)
petpt_numba_cpu(MSALB, SRAD, TMAX, TMIN, XHLAI)

MSALB = np.random.rand(10000000)
SRAD = np.random.randint(1, 100, size=10000000)
TMAX = np.random.randint(30, 40, size=10000000)
TMIN = np.random.randint(10, 15, size=10000000)
XHLAI = np.random.rand(10000000)
petpt_numpy_cpu(MSALB, SRAD, TMAX, TMIN, XHLAI)
petpt_numba_cpu(MSALB, SRAD, TMAX, TMIN, XHLAI)
EO = np.zeros_like(MSALB)
petpt_numba_gpu(MSALB, SRAD, TMAX, TMIN, XHLAI, EO)
print(f"The result of EO has shape: {EO.shape}")
print(f"The first element of EO is {EO[0]}")

# regular_matrix_comp()
# numpy_matrix_comp()
# numba_matrix_comp()
# numba_matrix_comp()
