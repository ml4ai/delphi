import inspect

import tangent
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
# from mpi4py.futures import MPIPoolExecutor

import delphi.program_analysis.autoTranslate.lambdas as lambdas


def visually_inspect_tangent():
    f = lambdas.CROP_YIELD__lambda__RAIN_0
    df = tangent.grad(f)
    code = inspect.getsource(df)
    print(code)


def perform_sensitivity_analysis():
    func = lambdas.CROP_YIELD__lambda__RAIN_0

    (args, _, _, _) = inspect.getargspec(func)

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


def evaluate(f, arg_matrix):
    return np.array([f(*args) for args in arg_matrix])


# visually_inspect_tangent()
perform_sensitivity_analysis()
