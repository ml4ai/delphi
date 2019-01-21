import inspect

import delphi.analysis.sensitivity.variance_methods as methods
from delphi.translators.for2py.data.PETPT import PETPT


def test_sobol_analysis():
    num_samples = 10000
    sig = inspect.signature(PETPT)
    num_args = len(list(sig.parameters))

    analyzer = methods.SobolAnalyzer(PETPT)
    analyzer.sample(num_samples=num_samples)

    expected_num_rows = num_samples * (2*num_args + 2)

    assert analyzer.samples.shape[0] == expected_num_rows
    assert analyzer.samples.shape[1] == num_args
