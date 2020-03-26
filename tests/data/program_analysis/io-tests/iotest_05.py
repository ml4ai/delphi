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


@static_vars([{'name': 'file_10', 'call': None, 'type': 'file_handle'}, {'name': 'file_20', 'call': None, 'type': 'file_handle'}])
def main():
    format_30: List[str] = [None]
    format_30 = ['/', "'f = '", 'f5.1', "'; i = '", 'i4']
    format_30_obj = Format(format_30)
    
    format_10: List[str] = [None]
    format_10 = ['2(i3,x,f5.2,x)']
    format_10_obj = Format(format_10)
    i: List[int] = [None]
    x: List[float] = [None]
    j: List[int] = [None]
    y: List[float] = [None]
    
    main.file_10 = open("infile3", "r")
    main.file_20 = open("outfile3", "w")
    
    (i[0], x[0], j[0], y[0],) = format_10_obj.read_line(main.file_10.readline())
    
    write_list_20 = [x[0], j[0]]
    write_line = format_30_obj.write_line(write_list_20)
    main.file_20.write(write_line)
    write_list_20 = [y[0], i[0]]
    write_line = format_30_obj.write_line(write_list_20)
    main.file_20.write(write_line)
    
    return

main()
