import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_30: List[str] = [None]
    format_30 = ['/', "'F = '", 'F5.1', "'; I = '", 'I4']
    format_30_obj = Format(format_30)
    
    format_10: List[str] = [None]
    format_10 = ['2(I3,X,F5.2,X)']
    format_10_obj = Format(format_10)
    i: List[int] = [None]
    x: List[float] = [None]
    j: List[int] = [None]
    y: List[float] = [None]
    
    file_10 = open("infile3", "r")
    file_20 = open("outfile3", "w")
    
    (i[0], x[0], j[0], y[0]) = format_10_obj.read_line(file_10.readline())
    
    write_list_20 = [x[0], j[0]]
    write_line = format_30_obj.write_line(write_list_20)
    file_20.write(write_line)
    write_list_20 = [y[0], i[0]]
    write_line = format_30_obj.write_line(write_list_20)
    file_20.write(write_line)
    
    return

main()
