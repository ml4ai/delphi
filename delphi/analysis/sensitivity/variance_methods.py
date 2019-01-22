from abc import ABCMeta, abstractmethod
from typing import Dict, Callable
import inspect

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np


class VarianceAnalyzer(metaclass=ABCMeta):
    """
    Meta-class for all variance based sensitivity analysis methods
    """

    def __init__(self, model, prob_def):
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

    def get_model_args(self):
        """Returns list of input arguments for the model."""
        return list(inspect.signature(self.model).parameters)

    def sample(self, num_samples: int=1000, second_order: bool=True) -> None:
        """
        Return a sample over the problem definition space using saltelli's
        sampling method. This sample will be analyzed to construct a
        sensitivity index.
        """
        print("Sampling over parameter bounds")
        self.samples = saltelli.sample(self.problem_definition,
                                       num_samples,
                                       calc_second_order=second_order)
        self.has_samples = True

    def evaluate(self) -> None:
        """Evaluate the model on the list of samples from self.sample()."""
        if not self.has_samples:
            raise RuntimeError("Attempted to evaluate model without samples")

        print("Evaluating samples")
        res = [self.model(*tuple([[a] for a in args])) for args in self.samples]
        self.outputs = np.array(res)
        self.has_outputs = True

    @abstractmethod
    def analyze(self) -> Dict:
        """Method to be implemented, should return a sensitivity index."""
        if not self.has_outputs:
            raise RuntimeError("Attempting analysis without outputs")

        print("Collecting sensitivity indices")


class SobolAnalyzer(VarianceAnalyzer):
    """
    Varianced-based method that constructs a Sobol sensitivity index
    """
    def __init__(self, model: Callable, prob_def):
        super().__init__(model, prob_def)

    def analyze(self, **kwargs) -> Dict:
        """
        Returns a sensitivity index as a dictionary with the following
        sensitivity indexes: S1, S2, ST.
        """
        super().analyze()
        return sobol.analyze(self.problem_definition, self.outputs, **kwargs)
