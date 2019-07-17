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


def myadd(arg1: List[int], arg2: List[int], arg3: List[int]):
    arg3[0] = (arg1[0] + arg2[0])
    

def main():
    format_10: List[str] = [None]
    format_10 = ['3(X,I3)']
    format_10_obj = Format(format_10)
    
    x: List[int] = [None]
    y: List[int] = [None]
    z: List[int] = [None]
    x[0] = 12
    y[0] = 13
    myadd(x, y, z)
    write_list_stream = [x[0], y[0], z[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    

main()
