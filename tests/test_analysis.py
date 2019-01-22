import inspect

import delphi.analysis.sensitivity.variance_methods as methods
from delphi.analysis.sensitivity.reporter import Reporter
from delphi.translators.for2py.data.PETPT import PETPT
from delphi.translators.for2py.data.PETASCE import PETASCE
from delphi.translators.for2py.data.Plant_pgm import MAIN


def test_PETPT():
    Ns = 100
    sig = inspect.signature(PETPT)
    num_args = len(list(sig.parameters))

    analyzer = methods.SobolAnalyzer(PETPT)
    analyzer.sample(num_samples=Ns)

    expected_num_rows = Ns * (2*num_args + 2)

    assert analyzer.samples.shape[0] == expected_num_rows
    assert analyzer.samples.shape[1] == num_args


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

    analyzer = methods.SobolAnalyzer(func, prob_def=problem)
    analyzer.sample(num_samples=100)
    analyzer.evaluate()
    Si = analyzer.analyze(parallel=False,
                          print_to_console=False,
                          n_processors=None)
    assert len(Si.keys()) == 6
    assert len(Si["S1"]) == len(args)


def test_PLANT_reporter():
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

    preset_vals = [[234], [33.9000015], [22.7999992], [1.0], [0.974399984],
                   [0.0], [12.0639181], [0.0]]

    bounds = [
        [0, 147],
        [0.000, 33.900],
        [-2.800, 21.100],
        [0.442, 1.000],
        [0.000, 1.000],
        [0.000, 15.000],
        [2.000, 12.039],
        [0.000, 0.100]
    ]

    func = PLANT_wrapper
    sig = inspect.signature(func)
    args = list(sig.parameters)
    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': bounds
    }

    analyzer = methods.SobolAnalyzer(func, prob_def=problem)
    reporter = Reporter(analyzer)
    analyzer.sample(num_samples=100)
    analyzer.evaluate()
    Si = analyzer.analyze(parallel=False,
                          print_to_console=False,
                          n_processors=None)
