import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random


@static_vars([{'name': 'w', 'call': [None], 'type': 'int'}])
def f(n: List[int], x: List[int]):
    if (n[0] == 0):
        f.w[0] = 111
    else:
        f.w[0] = int((2 * f.w[0]))
    x[0] = f.w[0]

@static_vars([{'name': 'w', 'call': [None], 'type': 'int'}])
def g(n: List[int], x: List[int]):
    if (n[0] == 0):
        g.w[0] = 999
    else:
        g.w[0] = int((g.w[0] / 3))
    x[0] = g.w[0]

def main():
    format_10: List[str] = [None]
    format_10 = ['"a = "', 'I5', '";   b = "', 'I5']
    format_10_obj = Format(format_10)
    
    a: List[int] = [None]
    b: List[int] = [None]
    f([0], a)
    g([0], b)
    
    write_list_stream = [a[0], b[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    f([1], a)
    g([1], b)
    write_list_stream = [a[0], b[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    f([1], a)
    g([1], b)
    write_list_stream = [a[0], b[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

main()
