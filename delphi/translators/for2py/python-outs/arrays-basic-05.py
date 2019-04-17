import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_10: List[str] = []
    format_10 = ['5(I5,X)']
    format_10_obj = Format(format_10)
    
    a = Array(int, [(-3, 1), (-4, 0)])
    i: List[int] = [0]
    j: List[int] = [0]
    for i[0] in range(-(3), 1+1):
        for j[0] in range(-(4), 0+1):
            a.set_((i[0], j[0]), (i[0] + j[0]))
    for i[0] in range(-(3), 1+1):
        write_list_stream = [a.get_((i[0], -4 )), a.get_((i[0], -3 )), a.get_((i[0], -2 )), a.get_((i[0], -1 )), a.get_((i[0], 0))]
        write_line = format_10_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    
    return

main()
