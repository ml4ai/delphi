import inspect
from tqdm import tqdm
import matplotlib.pyplot as plt

import delphi.analysis.sensitivity.variance_methods as methods
from delphi.translators.for2py.data.Plant_pgm import MAIN
from delphi.translators.for2py.data.PETASCE import PETASCE
from delphi.translators.for2py.data.PETPT import PETPT


def PLANT_wrapper(DOY, TMAX, TMIN, SWFAC1, PT, di, N, dN):
    SWFAC2 = [1.000]
    PD = [5.000]
    EMP1 = [0.104]
    EMP2 = [0.640]
    sla = [0.028]
    nb = [5.300]
    p1 = [0.030]

    return MAIN(DOY, TMAX, TMIN, SWFAC1, SWFAC2, PD,
                EMP1, EMP2, PT, sla, di, N, nb, dN, p1)


# bounds = [
#     [0, 147],
#     [0.000, 33.900],
#     [-2.800, 21.100],
#     [0.442, 1.000],
#     [0.000, 1.000],
#     [0.000, 15.000],
#     [2.000, 12.039],
#     [0.000, 0.100]
# ]
#
# func = PLANT_wrapper
# args = list(inspect.signature(func).parameters)
# analyzer = methods.SobolAnalyzer(func, prob_def={
#     'num_vars': len(args),
#     'names': args,
#     'bounds': bounds
# })

# def PETASCE_wrapper(CANHT, MSALB, SRAD, TDEW, TMAX, TMIN,
#                     WINDHT, WINDRUN, XHLAI, XLAT, XELEV):
#     DOY = [1]
#     MEEVP = ["A"]
#     return PETASCE(CANHT, DOY, MSALB, MEEVP, SRAD, TDEW, TMAX,
#                    TMIN, WINDHT, WINDRUN, XHLAI, XLAT, XELEV)
#
func = PETPT
sig = inspect.signature(func)
args = list(sig.parameters)
analyzer = methods.SobolAnalyzer(func, prob_def={
    'num_vars': len(args),
    'names': args,
    'bounds': [[.1, 1] for arg in args]
})

sample_amounts = list(range(10, 2010, 10))
# S1_low, S1_high = list(), list()
S1 = [[] for arg in args]
disp_length = len(sample_amounts)
for Ns in tqdm(sample_amounts, desc="Running SA"):
    analyzer.sample(num_samples=Ns)
    analyzer.evaluate()
    Si = analyzer.analyze(parallel=False,
                          print_to_console=False,
                          n_processors=None)
    s1 = [s for s in Si["S1"]]
    diffs = list()
    for idx, val in enumerate(s1):
        if len(S1[idx]) > 0:
            diffs.append(abs(val - S1[idx][-1]))

        S1[idx].append(val)

    # print(diffs)
    # if len(diffs) > 0 and all([d < 1e-4 for d in diffs]):
    #     disp_length = len(S1[0])
    #     break

    # S1_low.append(min(s1))
    # S1_high.append(max(s1))

plt.figure()
plt.title("S1 estimation given # of samples")
plt.xlabel("# of samples")
plt.ylabel("S1 value")
for idx, vals in enumerate(S1):
    if not all([v == 0.0 for v in vals]):
        plt.plot(sample_amounts[:disp_length], vals, label=args[idx])
plt.legend()
plt.show()
