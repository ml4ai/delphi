from typing import List, Callable
from numbers import Number, Real
from exceptions import EnvironmentError
import re

import numpy as np


UNSAFE_BUILTINS = re.compile(
    r"""
    (?=,)?(?=\s)?               # Match if proceeded by a comma or whitespace
    (
        __import__\([\"'][A-Za-z_]+[\"']\) |    # Match the import runtime var
        __loader__\. |    # Match the loader runtime var
        globals\(\) |                           # Match the global() builtin
        locals\(\)                              # Match the local() builtin
    )
    """,
    re.VERBOSE,
)
UNSAFE_IMPORT = r"\bimport [A-Za-z_]+\b"


class UnsafeOperationError(EnvironmentError):
    pass


class BadLambdaError(ValueError):
    pass


class BadDerivedTypeError(ValueError):
    pass


def load_lambda_function(func_str: str) -> Callable:
    # Checking to ensure the string has no executable import statements
    if re.search(UNSAFE_BUILTINS, func_str) is not None:
        raise UnsafeOperationError(f"found in lambda:\n{func_str}")

    # Checking for expected lambda expression header
    if re.match(r"lambda [A-Za-z0-9_,]*:", func_str) is None:
        raise RuntimeError(f"Unexpected form for lambda:\n{func_str}")

    try:
        func_ref = eval(func_str)

        # Checking to see if eval() produced a callable object
        if not isinstance(func_ref, callable):
            raise BadLambdaError(f"Callable not found for lambda:\n{func_str}")

        return func_ref
    except Exception as e:
        print(f"eval() failed for lambda: {func_str}")
        raise e


def load_derived_type(type_str: str) -> None:
    # Checking to ensure the string has no executable import statements
    bad_match = re.search(rf"({UNSAFE_BUILTINS})|({UNSAFE_IMPORT})", type_str)
    if bad_match is not None:
        raise UnsafeOperationError(f"found in derived-type:\n{type_str}")

    # Check for a dataclass structure match and extract the class name
    type_name_match = re.match(
        r"(?<=@dataclass\nclass )[A-Za-z_]+(?=:)", type_str
    )
    # Checking to see if the string starts with a dataclass
    if type_name_match is None:
        raise RuntimeError(f"Unexpected form for derived type:\n{type_str}")

    try:
        exec(type_str)

        # Checking to see if exec() produced the derived type class as a member
        # of the Python globals() object
        type_name = type_name_match.group()
        if type_name not in globals():
            raise BadDerivedTypeError(
                f"{type_name} not found for derived-type: {type_str}"
            )
    except Exception as e:
        print(f"exec() failed for derived-type: {type_str}")
        raise e
