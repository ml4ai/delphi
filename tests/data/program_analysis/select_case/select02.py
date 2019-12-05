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
    format_20: List[str] = [None]
    format_20 = ['A', 'I2', 'I4']
    format_20_obj = Format(format_20)
    
    format_30: List[str] = [None]
    format_30 = ['A']
    format_30_obj = Format(format_30)
    
    inc: List[int] = [None]
    y: List[int] = [None]
    for inc[0] in range(1, 10+1):
        
        if (inc[0] <= 3):
            y[0] = int((inc[0] * 2))
            write_list_stream = ["The variables I and Y have values: ", inc[0], y[0]]
            write_line = format_20_obj.write_line(write_list_stream)
            sys.stdout.write(write_line)
        else:
            if (inc[0] >= 9):
                y[0] = int((inc[0] * 3))
                write_list_stream = ["The variables I and Y have values: ", inc[0], y[0]]
                write_line = format_20_obj.write_line(write_list_stream)
                sys.stdout.write(write_line)
            else:
                if (inc[0] == 8):
                    y[0] = int((inc[0] * 4))
                    write_list_stream = ["The variables I and Y have values: ", inc[0], y[0]]
                    write_line = format_20_obj.write_line(write_list_stream)
                    sys.stdout.write(write_line)
                else:
                    if (inc[0] >= 4 and inc[0] <= 7):
                        y[0] = int((inc[0] * 5))
                        write_list_stream = ["The variables I and Y have values: ", inc[0], y[0]]
                        write_line = format_20_obj.write_line(write_list_stream)
                        sys.stdout.write(write_line)
                    else:
                        write_list_stream = ["Invalid Argument!"]
                        write_line = format_30_obj.write_line(write_list_stream)
                        sys.stdout.write(write_line)
    
    

main()
