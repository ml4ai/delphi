from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import numpy as np
import delphi.translators.for2py.math_ext as math

def select02__main__loop_0__assign__inc__0():
    return 1

def select02__main__loop_0__condition__IF_0__0(inc):
    return 0 <= inc < 11

def select02__main__loop_0__decision__EXIT__0(IF_0_0):
    return IF_0_0

def select02__main__loop_0__condition__IF_1__0(inc: int):
    return (inc <= 3)

def select02__main__loop_0__assign__y__0(inc: int):
    return int((inc*2))

def select02__main__loop_0__decision__y__1(y_0: int, y_1: int, IF_1_0: bool):
    return np.where(IF_1_0, y_1, y_0)

def select02__main__loop_0__condition__IF_1__1(inc: int):
    return (inc >= 9)

def select02__main__loop_0__assign__y__2(inc: int):
    return int((inc*3))

def select02__main__loop_0__decision__y__3(y_0: int, y_1: int, IF_1_1: bool):
    return np.where(IF_1_1, y_1, y_0)

def select02__main__loop_0__condition__IF_1__2(inc: int):
    return (inc == 8)

def select02__main__loop_0__assign__y__4(inc: int):
    return int((inc*4))

def select02__main__loop_0__decision__y__5(y_0: int, y_1: int, IF_1_2: bool):
    return np.where(IF_1_2, y_1, y_0)

def select02__main__loop_0__condition__IF_1__3(inc: int):
    return ((inc >= 4) and (inc <= 7))

def select02__main__loop_0__assign__y__6(inc: int):
    return int((inc*5))

def select02__main__loop_0__decision__y__7(y_0: int, y_1: int, IF_1_3: bool):
    return np.where(IF_1_3, y_1, y_0)

def select02__main__loop_0__assign_inc__1(inc):
    return inc + 1

