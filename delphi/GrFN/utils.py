from enum import Enum
import time


def timeit(method):
    """Timing wrapper for exectuion comparison."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"{method.__name__}:\t{((te - ts) * 1000):2.4f}ms")
        return result

    return timed


class ScopeNode(object):
    def __init__(self, container_dict, occ, parent=None):
        (_, namespace, scope, name) = container_dict["name"].split("::")
        self.name = f"{namespace}::{scope}::{name}::{occ}"
        self.body = container_dict["body"]
        self.repeat = container_dict["repeat"]
        self.arguments = container_dict["arguments"]
        self.updated = container_dict["updated"]
        self.returns = container_dict["return_value"]
        self.parent = parent
        self.variables = dict()  # NOTE: store base name as key and update index during wiring

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        vars_str = "\n".join([f"\t{k} -> {v}" for k, v in self.variables.items()])
        return f"{self.name}\nInputs: {self.inputs}\nVariables:\n{vars_str}"
