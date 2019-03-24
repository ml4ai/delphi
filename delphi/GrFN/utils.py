from enum import Enum
import time


def timeit(method):
    """Timing wrapper for exectuion comparison."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args)
        te = time.time()
        print(f"{method.__name__}:\t{((te - ts) * 1000):2.4f}ms")
        return result

    return timed


class NodeType(Enum):
    """Enum for the identification/tracking of a JSON function objects type."""
    CONTAINER = 0
    LOOP = 1
    ASSIGN = 2
    CONDITION = 3
    DECISION = 4
    VARIABLE = 5

    def is_function_node(self):
        return 2 <= self.value <= 4


def get_variable_name(var_dict, container_name):
    """Returns the unique node name of a variable."""
    return f"{container_name}::{var_dict['variable']}_{var_dict['index']}"


def get_node_type(type_str):
    """Returns the NodeType given a name of a JSON function object."""
    if type_str == "container":
        return NodeType.CONTAINER
    elif type_str == "loop_plate":
        return NodeType.LOOP
    elif type_str == "assign":
        return NodeType.ASSIGN
    elif type_str == "condition":
        return NodeType.CONDITION
    elif type_str == "decision":
        return NodeType.DECISION
    else:
        raise ValueError("Unrecognized type string: ", type_str)
