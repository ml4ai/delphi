import math
from delphi.translators.for2py.types_ext import Float32


def apply_op(num, op):
    if isinstance(num, Float32):
        return Float32(op(num._val))
    else:
        return op(num)


def cos(num):
    return apply_op(num, math.cos)


def sin(num):
    return apply_op(num, math.sin)


def tan(num):
    return apply_op(num, math.tan)


def acos(num):
    return apply_op(num, math.acos)


def exp(num):
    return apply_op(num, math.exp)


def log(num):
    return apply_op(num, math.log)


def sqrt(num):
    return apply_op(num, math.sqrt)
