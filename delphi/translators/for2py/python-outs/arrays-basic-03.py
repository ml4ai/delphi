import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_10: List[str] = []
    format_10 = ['I5']
    format_10_obj = Format(format_10)
    
    array = Array(int, [(1, 10)])
    idx = Array(int, [(1, 10)])
    i: List[int] = [0]
    for i[0] in range(1, 10+1):
        array.set_((i[0]), (i[0] * i[0]))
    for i[0] in range(1, 5+1):
        idx.set_((i[0]), (2 * i[0]))
        idx.set_((i[0] + 5), ((2 * i[0]) - 1))
    for i[0] in range(1, 10+1):
        write_list_stream = [array.get_((idx.get_((i[0]))))]
        write_line = format_10_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    
    return

main()
