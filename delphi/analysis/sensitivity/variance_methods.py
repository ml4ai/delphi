from abc import ABCMeta, abstractmethod
import inspect

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np


class VarianceAnalyzer(metaclass=ABCMeta):
    """
    Meta-class for all variance based sensitivity analysis methods
    """

    def __init__(self, model, prob_def=None):
        self.has_samples = False
        self.has_outputs = False
        self.model = model

        if prob_def is None:
            sig = inspect.signature(self.model)
            args = list(sig.parameters)

            self.problem_definition = {
                'num_vars': len(args),
                'names': args,
                'bounds': [[-100, 100] for arg in args]
            }
        else:
            self.problem_definition = prob_def

    def sample(self, num_samples=1000, second_order=True):
        print("Sampling over parameter bounds")
        self.samples = saltelli.sample(self.problem_definition,
                                       num_samples,
                                       calc_second_order=second_order)
        self.has_samples = True

    def evaluate(self):
        if not self.has_samples:
            raise RuntimeError("Attempted to evaluate model without samples")

        print("Evaluating samples")
        res = [self.model(*tuple([[a] for a in args])) for args in self.samples]
        self.outputs = np.array(res)
        self.has_outputs = True

    @abstractmethod
    def analyze(self):
        if not self.has_outputs:
            raise RuntimeError("Attempting analysis without outputs")

        print("Collecting sensitivity indices")


class SobolAnalyzer(VarianceAnalyzer):
    def __init__(self, model, prob_def=None):
        super().__init__(model, prob_def=prob_def)

    def analyze(self, **kwargs):
        super().analyze()
        return sobol.analyze(self.problem_definition, self.outputs, **kwargs)
