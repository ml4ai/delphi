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


def odd_number():
    format_10: List[str] = [None]
    format_10 = ["'k = '", 'i3', "'; n = '", 'i8']
    format_10_obj = Format(format_10)
    
    n: List[int] = [None]
    k: List[int] = [None]
    n[0] = 19
    k[0] = 0
    while True:
        k[0] = (k[0] + 1)
        if (k[0] != 5):
            if (k[0] <= n[0]):
                if ((k[0] % 2) != 0):
                    write_list_stream = [k[0], n[0]]
                    write_line = format_10_obj.write_line(write_list_stream)
                    sys.stdout.write(write_line)
            else:
                break
    

odd_number()
