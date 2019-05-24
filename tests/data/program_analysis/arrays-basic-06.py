import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_10: List[str] = [None]
    format_10 = ['5(I5,X)']
    format_10_obj = Format(format_10)
    
    a = Array(int, [(-(3), 1), (0, 5), (10, 14)])
    i: List[int] = [None]
    j: List[int] = [None]
    k: List[int] = [None]
    for i[0] in range(-(3), 1+1):
        for j[0] in range(1, 5+1):
            for k[0] in range(10, 14+1):
                a.set_((i[0], j[0], k[0]), ((i[0] + j[0]) + k[0]))
    for i[0] in range(-(3), 1+1):
        for j[0] in range(1, 5+1):
            write_list_stream = [a.get_((i[0], j[0], 10)), a.get_((i[0], j[0], 11)), a.get_((i[0], j[0], 12)), a.get_((i[0], j[0], 13)), a.get_((i[0], j[0], 14))]
            write_line = format_10_obj.write_line(write_list_stream)
            sys.stdout.write(write_line)
    
    return

main()
