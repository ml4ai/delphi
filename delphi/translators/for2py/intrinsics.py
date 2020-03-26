import math
import numpy as np
from numpy import ndarray
from functools import singledispatch
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from delphi.translators.for2py.types_ext import Float32


@singledispatch
def nint(element):
    """Rounds the number to the nearest integer value.
    Passed element argument can be integer, real, Array,
    or list. Depends on the types of passed element, the
    return type may vary.
    """
    raise TypeError(f"<nint> unhandled type: {type(element)}")


@nint.register
def _(element: float):
    return round_value(element)


@nint.register
def _(element: int):
    return element


@nint.register
def _(element: list):
    new_list = []
    for elem in element:
        if isinstance(elem, float):
            new_list.append(round_value(elem))
        elif isinstance(elem, list):
            new_list.append(nint(elem))
        else:
            new_list.append(elem)
    return new_list


@nint.register
def _(element: ndarray):
    """
    Handles Fortran NINT style rounding of floats in an NDarray. Whether the
    number is positive or negative, the integer portion of the number is
    increased (without regard to sign) if the decimal portion of the number
    is >= 0.5
    """
    return np.where(
        (element >= 0) ^ (np.modf(np.absolute(element))[0] >= 0.5),
        np.floor(element),
        np.ceil(element),
    )


@nint.register
def _(element: Array):
    arr_bounds = element.bounds()
    low = arr_bounds[0][0] + 1
    up = arr_bounds[0][1] + 1

    new_array = Array(element.get_type(), arr_bounds)
    for idx in range(low, up):
        arr_element = element.get_(idx)
        # Multi-dimensional array.
        # TODO: Currently handle only 2D arrays.
        if type(arr_element) == list:
            for idx2 in range(1, len(arr_element)):
                rounded_elm = nint(arr_element[idx2])
                new_array.set_((idx, idx2), rounded_elm)
        else:
            new_array.set_((idx), nint(arr_element))

    return new_array


def round_value(element: float):
    """
    Handles Fortran NINT style rounding of a floating point value. Whether the
    number is positive or negative, the integer portion of the number is
    increased (without regard to sign) if the decimal portion of the number
    is >= 0.5
    """
    return (
        math.floor(element)
        if (element >= 0) ^ (math.modf(abs(element))[0] >= 0.5)
        else math.ceil(element)
    )
