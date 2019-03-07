import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from ctypes import c_int, c_float, c_wchar_p


def main():
    format_10: c_wchar_p = c_wchar_p()
    format_10 = ['5(I5,X)']
    format_10_obj = Format(format_10)
    
    a = Array(int, [(-3, 1), (1, 5), (10, 14)])
    i: c_int = c_int(0)
    j: c_int = c_int(0)
    k: c_int = c_int(0)
    for i.value in range(-(3), 1+1):
        for j.value in range(1, 5+1):
            for k.value in range(10, 14+1):
                a.set_((i.value, j.value, k.value), ((i.value + j.value) + k.value))
    for i.value in range(-(3), 1+1):
        for j.value in range(1, 5+1):
            write_list_stream = [a.get_((i.value, j.value, 10)), a.get_((i.value, j.value, 11)), a.get_((i.value, j.value, 12)), a.get_((i.value, j.value, 13)), a.get_((i.value, j.value, 14))]
            write_line = format_10_obj.write_line(write_list_stream)
            sys.stdout.write(write_line)
    
    return

main()
