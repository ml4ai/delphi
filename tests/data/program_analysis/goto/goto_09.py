import sys
import os
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from delphi.translators.for2py import intrinsics
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random


def factorial():
    format_10: List[str] = [None]
    format_10 = ["'i = '", 'i3', "'; fact = '", 'i8']
    format_10_obj = Format(format_10)
    
    i: List[int] = [None]
    n: List[int] = [None]
    fact: List[int] = [None]
    i[0] = 0
    n[0] = 10
    fact[0] = 0
    for i[0] in range(1, n[0]+1):
        if (i[0] < 20):
            if (i[0] == 1):
                fact[0] = (fact[0] + 1)
                write_list_stream = [i[0], fact[0]]
                write_line = format_10_obj.write_line(write_list_stream)
                sys.stdout.write(write_line)
            else:
                if (i[0] <= 10):
                    fact[0] = int((fact[0] * i[0]))
                    write_list_stream = [i[0], fact[0]]
                    write_line = format_10_obj.write_line(write_list_stream)
                    sys.stdout.write(write_line)
                else:
                    break
    goto_flag_1: List[bool] = [None]
    goto_flag_1[0] = True
    if goto_flag_1[0]:
        return
    

factorial()
