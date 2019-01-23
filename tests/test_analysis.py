import inspect

import delphi.analysis.sensitivity.variance_methods as methods
from delphi.analysis.sensitivity.reporter import Reporter
from delphi.translators.for2py.data.PETASCE import PETASCE
from delphi.translators.for2py.data.Plant_pgm import MAIN


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

    Ns = 100
    num_args = len(args)
    analyzer = methods.SobolAnalyzer(func, prob_def=problem)
    analyzer.sample(num_samples=Ns)
    analyzer.evaluate()
    Si = analyzer.analyze(parallel=False,
                          print_to_console=False,
                          n_processors=None)
    assert len(Si.keys()) == 6
    assert len(Si["S1"]) == len(args)

    expected_num_rows = Ns * (2*num_args + 2)
    assert analyzer.samples.shape[0] == expected_num_rows
    assert analyzer.samples.shape[1] == num_args


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

    num_samples = 100
    func = PLANT_wrapper
    args = list(inspect.signature(func).parameters)
    reporter = Reporter(methods.SobolAnalyzer(func, prob_def={
        'num_vars': len(args),
        'names': args,
        'bounds': bounds
    }), num_samples=num_samples)

    (min_val, min_idx1, min_idx2) = reporter.get_min_s2_sensitivity()
    (max_val, max_idx1, max_idx2) = reporter.get_max_s2_sensitivity()

    assert 0 <= min_idx1 <= len(bounds)
    assert 0 <= min_idx2 <= len(bounds)
    assert 0 <= max_idx1 <= len(bounds)
    assert 0 <= max_idx2 <= len(bounds)
    assert min_val < max_val

    surface_bounds = ((bounds[max_idx1]), (bounds[max_idx2]))
    preset_vals = [[234], [33.9000015],
                   [22.7999992], [1.0],
                   [0.974399984], [0.0],
                   [12.0639181], [0.0]]
    (X, Y, surface) = reporter.get_sensitivity_surface(max_idx1,
                                                       max_idx2,
                                                       surface_bounds,
                                                       preset_vals)

    assert surface.shape == (num_samples, num_samples)
    assert bounds[max_idx1][0] <= min(X[0]) and max(X[0]) <= bounds[max_idx1][1]
    assert bounds[max_idx2][0] <= min(Y[0]) and max(Y[0]) <= bounds[max_idx2][1]

    # Ghost testing visualization code
    plot = reporter.visualize_surface(max_idx1, max_idx2, X, Y, surface)
    assert True
