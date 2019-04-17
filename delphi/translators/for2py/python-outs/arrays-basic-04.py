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
    
    arr = Array(int, [(1, 5), (1, 5)])
    i: List[int] = [0]
    j: List[int] = [0]
    file_10 = open("OUTPUT", "w")
    for i[0] in range(1, 5+1):
        for j[0] in range(1, 5+1):
            arr.set_((i[0], j[0]), (i[0] + j[0]))
    for i[0] in range(1, 5+1):
        write_list_10 = [arr.get_((i[0], 1)), arr.get_((i[0], 2)), arr.get_((i[0], 3)), arr.get_((i[0], 4)), arr.get_((i[0], 5))]
        write_line = format_10_obj.write_line(write_list_10)
        file_10.write(write_line)
    
    return

main()
