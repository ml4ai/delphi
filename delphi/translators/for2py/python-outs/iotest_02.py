import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_10: List[str] = []
    format_10 = ['I5', 'I5']
    format_10_obj = Format(format_10)
    i: List[int] = [0]
    j: List[int] = [0]
    
    file_10 = open("infile1", "r")
    file_20 = open("outfile1", "w")
    (i[0], j[0]) = format_10_obj.read_line(file_10.readline())
    
    write_list_20 = [j[0], i[0]]
    write_line = format_10_obj.write_line(write_list_20)
    file_20.write(write_line)
    
    return

main()
