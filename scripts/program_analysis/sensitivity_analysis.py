import inspect
import pickle
import time

import tangent
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from mpi4py.futures import MPIPoolExecutor

# import delphi.program_analysis.autoTranslate.lambdas as lambdas
import delphi.program_analysis.autoTranslate.PETPT as lambdas
import delphi.program_analysis.data.PETASCE as lambdas
import delphi.program_analysis.data.Plant_pgm as plant


def visually_inspect_tangent():
    def test_func(x, y):
        ans = 0
        for i in range(y):
            z = x**3 + 3*x**2 + 6
            ans += max(z, 1)
        return ans
    # f = lambdas.CROP_YIELD__lambda__RAIN_0
    # f = lambdas.PETASCE
    f = test_func
    df = tangent.grad(f)
    code = inspect.getsource(df)
    print(code)


def perform_sensitivity_analysis():
    func = lambdas.CROP_YIELD__lambda__RAIN_0

    sig = inspect.signature(func)
    args = list(sig.parameters)

    param_def = {
        'num_vars': len(args),
        'names': args,
        'bounds': [[0, 30], [0, 1], [0, 100], [0, 1]]
    }

    print("Sampling over parameter bounds")
    param_values = saltelli.sample(param_def, 10000, calc_second_order=True)

    print("Evaluating samples")
    y = evaluate(func, param_values)

    print("Collecting sensitivity indices")
    Si = sobol.analyze(param_def, y, parallel=False, print_to_console=False, n_processors=None)
    print(Si)


def PETPT_SA():
    func = lambdas.PETPT
    sig = inspect.signature(func)
    args = list(sig.parameters)

    param_def = {
        'num_vars': len(args),
        'names': args,
        'bounds': [[-100, 100] for arg in args]
    }

    print("Sampling over parameter bounds")
    param_values = saltelli.sample(param_def, 100, calc_second_order=True)

    print("Evaluating samples")
    y = evaluate_PET(func, param_values)

    print("Collecting sensitivity indices")
    Si = sobol.analyze(param_def, y, parallel=False, print_to_console=False, n_processors=None)
    print(Si)


def PETASCE_SA():
    func = PETASCE_wrapper
    sig = inspect.signature(func)
    args = list(sig.parameters)

    param_def = {
        'num_vars': len(args),
        'names': args,
        'bounds': [[.1, 1] for arg in args]
    }

    print("Sampling over parameter bounds")
    param_values = saltelli.sample(param_def, 10000, calc_second_order=True)

    print("Evaluating samples")
    y = evaluate_PET(func, param_values)

    print("Collecting sensitivity indices")
    Si = sobol.analyze(param_def, y, parallel=False, print_to_console=False, n_processors=None)
    print(Si)


def PLANT_SA():
    func = PLANT_wrapper
    sig = inspect.signature(func)
    args = list(sig.parameters)

    param_def = {
        'num_vars': len(args),
        'names': args,
        'bounds': [
            [0, 147],
            [0.000, 33.900],
            [-2.800, 21.100],
            [0.442, 1.000],
            [0.000, 1.000],
            [0.000, 15.000],
            [2.000, 12.039],
            [0.000, 0.100],
        ]
    }

    print("Sampling over parameter bounds")
    param_values = saltelli.sample(param_def, 1000000, calc_second_order=True)

    print("Evaluating samples")
    y = evaluate_PET(func, param_values)

    print("Collecting sensitivity indices")
    Si = sobol.analyze(param_def, y, parallel=False, print_to_console=False, n_processors=None)
    # print(Si)
    matprint(Si["S2"])

    data = {
        "samples": param_values,
        "outputs": y,
        "Si": Si
    }

    pickle.dump(data, open("PLANT-sensitivity.pkl", "wb"))


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def sa_sample_test():
    full_times = list()
    sobol_times = list()

    func = test_func_5
    sig = inspect.signature(func)
    args = list(sig.parameters)

    param_def = {
        'num_vars': len(args),
        'names': args,
        'bounds': [[-100, 100] for arg in args]
    }
    # sample_amounts = [10, 100, 1000, 10000, 100000, 1000000]
    sample_amounts = list(range(10000, 90000, 10000))
    for num_samples in sample_amounts:
        start_time = time.time()
        param_values = saltelli.sample(param_def, num_samples, calc_second_order=True)
        y = evaluate(func, param_values)
        start_sobol = time.time()
        Si = sobol.analyze(param_def, y, parallel=False, print_to_console=False, n_processors=None)

        sobol_time = time.time() - start_sobol
        full_time = time.time() - start_time
        full_times.append(full_time)
        sobol_times.append(sobol_time)
        print("Total time for {}:\t{}".format(num_samples, full_time))

    plt.figure()
    plt.plot(sample_amounts, sobol_times, color="blue")
    plt.plot(sample_amounts, full_times, color="red")
    plt.title("SA time as a function of sample size")
    plt.xlabel("# of samples")
    plt.ylabel("Processing time")
    plt.legend(["Sobol time", "Full time"])
    plt.show()


def sa_input_test():
    funcs = [test_func_1, test_func_2, test_func_3, test_func_4, test_func_5, test_func_6, test_func_7, test_func_8, test_func_9, test_func_10]
    full_times = list()
    sobol_times = list()
    for func in funcs:
        start_time = time.time()
        sig = inspect.signature(func)
        args = list(sig.parameters)

        param_def = {
            'num_vars': len(args),
            'names': args,
            'bounds': [[-100, 100] for arg in args]
        }

        param_values = saltelli.sample(param_def, 10000, calc_second_order=True)
        y = evaluate(func, param_values)
        start_sobol = time.time()
        Si = sobol.analyze(param_def, y, parallel=False, print_to_console=False, n_processors=None)

        sobol_time = time.time() - start_sobol
        full_time = time.time() - start_time
        full_times.append(full_time)
        sobol_times.append(sobol_time)
        print("Total time for {}:\t{}".format(func.__name__, full_time))

    num_args = list(range(2, 12))
    plt.figure()
    plt.plot(num_args, sobol_times, color="blue")
    plt.plot(num_args, full_times, color="red")
    plt.title("SA time as a function of # of inputs")
    plt.xlabel("# of inputs")
    plt.ylabel("Processing time")
    plt.legend(["Sobol time", "Full time"])
    plt.show()


def PETASCE_wrapper(CANHT, MSALB, SRAD, TDEW, TMAX, TMIN, WINDHT, WINDRUN,
                    XHLAI, XLAT, XELEV):
    DOY = [1]
    MEEVP = ["A"]
    return lambdas.PETASCE(CANHT, DOY, MSALB, MEEVP, SRAD, TDEW, TMAX, TMIN,
                           WINDHT, WINDRUN, XHLAI, XLAT, XELEV)


def PLANT_wrapper(DOY, TMAX, TMIN, SWFAC1, PT, di, N, dN):
    SWFAC2 = [1.000]
    PD = [5.000]
    EMP1 = [0.104]
    EMP2 = [0.640]
    sla = [0.028]
    nb = [5.300]
    p1 = [0.030]

    return plant.MAIN(DOY, TMAX, TMIN, SWFAC1, SWFAC2, PD, EMP1, EMP2, PT, sla,
                      di, N, nb, dN, p1)


def evaluate_PET(f, arg_matrix):
    return np.array([f(*tuple([[arg] for arg in args])) for args in arg_matrix])


def evaluate(f, arg_matrix):
    return np.array([f(*args) for args in arg_matrix])


def test_func_1(a, b):
    return (a + b) + (a * b)


def test_func_2(a, b, c):
    return (a + b + c) + (a * b * c)


def test_func_3(a, b, c, d):
    return (a + b + c + d) + (a * b * c * d)


def test_func_4(a, b, c, d, e):
    return (a + b + c + d + e) + (a * b * c * d * e)


def test_func_5(a, b, c, d, e, f):
    return (a + b + c + d + e + f) + (a * b * c * d * e * f)


def test_func_6(a, b, c, d, e, f, g):
    return (a + b + c + d + e + f + g) + (a * b * c * d * e * f * g)


def test_func_7(a, b, c, d, e, f, g, h):
    return (a + b + c + d + e + f + g + h) + (a * b * c * d * e * f * g * h)


def test_func_8(a, b, c, d, e, f, g, h, i):
    return (a + b + c + d + e + f + g + h + i) + \
           (a * b * c * d * e * f * g * h * i)


def test_func_9(a, b, c, d, e, f, g, h, i, j):
    return (a + b + c + d + e + f + g + h + i + j) + \
           (a * b * c * d * e * f * g * h * i * j)


def test_func_10(a, b, c, d, e, f, g, h, i, j, k):
    return (a + b + c + d + e + f + g + h + i + j + k) + \
           (a * b * c * d * e * f * g * h * i * j + k)


# visually_inspect_tangent()
# perform_sensitivity_analysis()
# PETPT_SA()
# sa_input_test()
# sa_sample_test()
# PETASCE_SA()
PLANT_SA()
