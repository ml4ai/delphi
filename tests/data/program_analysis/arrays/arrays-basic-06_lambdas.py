from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import numpy as np
from delphi.translators.for2py import intrinsics
import delphi.translators.for2py.math_ext as math

def arrays_basic_06__main__assign__a__0():
    a = [[0] * (1 - -(3)), [0] * (5 - 0), [0] * (14 - 10)]
    return a

def arrays_basic_06__main__loop_0__assign__i__0():
    return 3

def arrays_basic_06__main__loop_0__condition__IF_0__0(i):
    return 0 <= i < 2

def arrays_basic_06__main__loop_0__loop_1__assign__j__0():
    return 1

def arrays_basic_06__main__loop_0__loop_1__condition__IF_0__0(j):
    return 0 <= j < 6

def arrays_basic_06__main__loop_0__loop_1__loop_2__assign__k__0():
    return 10

def arrays_basic_06__main__loop_0__loop_1__loop_2__condition__IF_0__0(k):
    return 0 <= k < 15

def arrays_basic_06__main__loop_0__loop_1__loop_2__assign__a_ijk__0(a, i: int, j: int, k: int):
    a[i][j][k] = int(((i+j)+k))
    return a

def arrays_basic_06__main__loop_0__loop_1__loop_2__decision__EXIT__0(IF_0_0):
    return (not IF_0_0)

def arrays_basic_06__main__loop_0__loop_1__loop_2__assign_k__1(k):
    return k + 1

def arrays_basic_06__main__loop_0__loop_1__decision__EXIT__0(IF_0_0):
    return (not IF_0_0)

def arrays_basic_06__main__loop_0__loop_1__assign_j__1(j):
    return j + 1

def arrays_basic_06__main__loop_0__decision__EXIT__0(IF_0_0):
    return (not IF_0_0)

def arrays_basic_06__main__loop_0__assign_i__1(i):
    return i + 1

def arrays_basic_06__main__loop_3__assign__i__0():
    return 3

def arrays_basic_06__main__loop_3__condition__IF_0__0(i):
    return 0 <= i < 2

def arrays_basic_06__main__loop_3__loop_4__assign__j__0():
    return 1

def arrays_basic_06__main__loop_3__loop_4__condition__IF_0__0(j):
    return 0 <= j < 6

def arrays_basic_06__main__loop_3__loop_4__decision__EXIT__0(IF_0_0):
    return (not IF_0_0)

def arrays_basic_06__main__loop_3__loop_4__assign_j__1(j):
    return j + 1

def arrays_basic_06__main__loop_3__decision__EXIT__0(IF_0_0):
    return (not IF_0_0)

def arrays_basic_06__main__loop_3__assign_i__1(i):
    return i + 1

