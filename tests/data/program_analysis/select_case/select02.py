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


def main():
    format_10: List[str] = [None]
    format_10 = ['A', 'I2', 'I2', 'I4']
    format_10_obj = Format(format_10)
    
    format_20: List[str] = [None]
    format_20 = ['A']
    format_20_obj = Format(format_20)
    
    i: List[int] = [5]
    x: List[int] = [40]
    y: List[int] = [None]
    z: List[int] = [2]
    
    if (i[0] <= 3):
        y[0] = int((x[0] / 4))
        write_list_stream = ["The variable is I, A, and Y are: ", i[0], y[0], (y[0] * z[0])]
        write_line = format_10_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    else:
        if (i[0] >= 9):
            y[0] = int((x[0] / 10))
            write_list_stream = ["The variable is I, A, and Y are: ", i[0], y[0], (y[0] * z[0])]
            write_line = format_10_obj.write_line(write_list_stream)
            sys.stdout.write(write_line)
        else:
            if (i[0] == 8):
                y[0] = int((x[0] / 2))
                write_list_stream = ["The variable is I, A, and Y are: ", i[0], y[0], (y[0] * z[0])]
                write_line = format_10_obj.write_line(write_list_stream)
                sys.stdout.write(write_line)
            else:
                if (i[0] >= 4 and i[0] <= 7):
                    y[0] = int((x[0] / 8))
                    write_list_stream = ["The variable is I, A, and Y are: ", i[0], y[0], (y[0] * z[0])]
                    write_line = format_10_obj.write_line(write_list_stream)
                    sys.stdout.write(write_line)
                else:
                    write_list_stream = ["Invalid Argument!"]
                    write_line = format_20_obj.write_line(write_list_stream)
                    sys.stdout.write(write_line)
                
                

main()
