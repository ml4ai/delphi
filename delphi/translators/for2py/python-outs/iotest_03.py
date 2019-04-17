import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_10: List[str] = []
    format_10 = ['F5.3']
    format_10_obj = Format(format_10)
    x: List[float] = [0.0]
    
    file_10 = open("infile2", "r")
    file_20 = open("outfile2", "w")
    (x[0]) = format_10_obj.read_line(file_10.readline())
    
    write_list_20 = [x[0]]
    write_line = format_10_obj.write_line(write_list_20)
    file_20.write(write_line)
    
    return

main()
