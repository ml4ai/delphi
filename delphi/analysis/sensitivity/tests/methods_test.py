import inspect
import pickle

from delphi.program_analysis.sensitivity.VarianceMethods import SobolAnalyzer

from delphi.program_analysis.data.PETPT import PETPT
from delphi.program_analysis.data.PETASCE import PETASCE
from delphi.program_analysis.data.Plant_pgm import MAIN


def test_PETPT():
    analyzer = SobolAnalyzer(PETPT)
    analyzer.sample(num_samples=100)
    analyzer.evaluate()
    Si = analyzer.analyze(parallel=False,
                          print_to_console=False,
                          n_processors=None)
    print(Si)


def test_PETASCE():
    def PETASCE_wrapper(CANHT, MSALB, SRAD, TDEW, TMAX, TMIN,
                        WINDHT, WINDRUN, XHLAI, XLAT, XELEV):
        DOY = [1]
        MEEVP = ["A"]
        return PETASCE(CANHT, DOY, MSALB, MEEVP, SRAD, TDEW, TMAX,
                       TMIN, WINDHT, WINDRUN, XHLAI, XLAT, XELEV)

    func = PETASCE_wrapper
    sig = inspect.signature(func)
    args = list(sig.parameters)
    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [[.1, 1] for arg in args]
    }

    analyzer = SobolAnalyzer(func, prob_def=problem)
    analyzer.sample(num_samples=10000)
    analyzer.evaluate()
    Si = analyzer.analyze(parallel=False,
                          print_to_console=False,
                          n_processors=None)
    print(Si)


def test_PLANT():
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

    func = PLANT_wrapper
    sig = inspect.signature(func)
    args = list(sig.parameters)
    problem = {
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

    analyzer = SobolAnalyzer(func, prob_def=problem)
    analyzer.sample(num_samples=10000)
    analyzer.evaluate()
    Si = analyzer.analyze(parallel=False,
                          print_to_console=False,
                          n_processors=None)
    matprint(Si["S2"])
    data = {"samples": problem, "outputs": analyzer.outputs, "Si": Si}
    pickle.dump(data, open("PLANT-sensitivity.pkl", "wb"))


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col])
                 for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


# test_PETPT()
# test_PETASCE()
test_PLANT()
