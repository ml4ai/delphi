from typing import List
import math
from fortran_format import *

def MAIN():
    format_30: List[str] = []
    format_30 = ['/', "'F = '", 'F5.1', "'; I = '", 'I4']
    format_30_obj = Format(format_30)
    
    format_10: List[str] = []
    format_10 = ['2(I3,X,F5.2,X)']
    format_10_obj = Format(format_10)
    I: List[int] = [0]
    X: List[float] = [0.0]
    J: List[int] = [0]
    Y: List[float] = [0.0]
    
    file_10 = open("infile3", "r")
    file_20 = open("outfile3", "w")
    
    (I,X,J,Y,) = format_10_obj.read_line(file_10.readline())
    write_list_20 = [X, J]
    write_line = format_30_obj.write_line(write_list_20)
    file_20.write(write_line)
    write_list_20 = [Y, I]
    write_line = format_30_obj.write_line(write_list_20)
    file_20.write(write_line)
    
    return

MAIN()
