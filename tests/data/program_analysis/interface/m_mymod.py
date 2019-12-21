import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random


def foo (arg1=None, arg2=None):
    num_passed_args = 0
    if arg1 != None:
        num_passed_args += 1
    if arg2 != None:
        num_passed_args += 1

    if num_passed_args == 2:
        if isinstance(arg1[0], int) and isinstance(arg2[0], int):
            foo_int(arg1, arg2)
        if isinstance(arg1[0], float) and isinstance(arg2[0], int):
            foo_real(arg1, arg2)
        if isinstance(arg1[0], bool) and isinstance(arg2[0], int):
            foo_bool(arg1, arg2)

def foo_int(x: List[int], result: List[int]):
    result[0] = int(((47 * x[0]) + 23))
    

def foo_real(x: List[Real], result: List[int]):
    result[0] = int(((int(x[0]) * 31) + 17))
    

def foo_bool(x: List[bool], result: List[int]):
    if x[0]:
        result[0] = 937
    else:
        result[0] = -(732)
    