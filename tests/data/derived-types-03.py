import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from ctypes import c_int, c_float, c_wchar_p

@dataclass
class mytype_1:
    a: int = None
    b: float = None

@dataclass
class mytype_2:
    a: int = None
    b: float = None


def main():
    format_10: c_wchar_p = c_wchar_p()
    format_10 = ['I5', '3X', 'F7.3']
    format_10_obj = Format(format_10)
    
    x =  mytype_1()
    y =  mytype_2()
    x.a = 123
    x.b = 4.56
    y.a = (x.a * 2)
    y.b = (x.b * 3)
    
    write_list_stream = [x.a, x.b]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = [y.a, y.b]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

main()
