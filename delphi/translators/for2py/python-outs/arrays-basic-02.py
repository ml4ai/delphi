import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_10: List[str] = []
    format_10 = ['I5', 'X', 'I5']
    format_10_obj = Format(format_10)
    
    array = Array(int, [(-5, 5)])
    i: List[int] = [0]
    for i[0] in range(-(5), 5+1):
        array.set_((i[0]), (i[0] * i[0]))
    for i[0] in range(-(5), 5+1):
        write_list_stream = [i[0], array.get_((i[0]))]
        write_line = format_10_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    
    return

main()
