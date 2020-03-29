from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import numpy as np
from delphi.translators.for2py import intrinsics
import delphi.translators.for2py.math_ext as math

def select02__main__loop_0__assign__inc__0():
    return 1

def select02__main__loop_0__condition__IF_0__0(inc):
    return 0 <= inc < 11

def select02__main__loop_0__IF_0__condition__COND_0__0(inc: int):
    return ((inc >= "-inf") and (inc <= 3))

def select02__main__loop_0__IF_0__assign__y__0(inc: int):
    return int((inc*2))

def select02__main__loop_0__IF_0__condition__COND_1__0(inc: int):
    return ((inc >= 9) and (inc <= "inf"))

def select02__main__loop_0__IF_0__assign__y__1(inc: int):
    return int((inc*3))

def select02__main__loop_0__IF_0__condition__COND_2__0(inc: int):
    return (inc == 8)

def select02__main__loop_0__IF_0__assign__y__2(inc: int):
    return int((inc*4))

def select02__main__loop_0__IF_0__condition__COND_3__0(inc: int):
    return ((inc >= 4) and (inc <= 7))

def select02__main__loop_0__IF_0__assign__y__3(inc: int):
    return int((inc*5))

def select02__main__loop_0__decision__EXIT__0(IF_0_0):
    return (not IF_0_0)

def select02__main__loop_0__assign_inc__1(inc):
    return inc + 1

