import csv
import json
import time
import functools
from typing import Iterable


def timeit(func):
    """Record the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        return (value, end_time - start_time)
    return wrapper_timer


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


def results_to_csv(
    filepath: str, inputMat: Iterable, outputMat: Iterable
) -> None:
    # TODO khan: implement this such that each row of a csv file has a set of
    # inputs followed by a set of outputs
    return NotImplemented


def results_to_json(
    filepath: str, inputMat: Iterable, outputMat: Iterable
) -> None:
    # TODO khan: implement this such that each row of a json file as an object
    # with the keys (input_data, output_data). Store the corresponding arrays as
    # lists at those outputs
    return NotImplemented


def bounds_from_csv(filepath: str) -> dict:
    # TODO khan: Implement bound loading from a CSV file here
    return NotImplemented


def bounds_from_json(filepath: str) -> dict:
    # TODO khan: Implement bound loading from a JSON file here
    return NotImplemented
