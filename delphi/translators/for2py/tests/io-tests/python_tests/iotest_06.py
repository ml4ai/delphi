from typing import List
import math
from fortran_format import *

def MAIN():
    format_10 = ['F5.2', 'X', 'F5.2']
    format_10_obj = Format(format_10)
    
    format_11 = ["'The values of X and Y are: '"]
    format_11_obj = Format(format_11)
    
    format_12 = ['F6.3', '3X', 'F4.2']
    format_12_obj = Format(format_12)
    
    X: List[float] = [0.0]
    Y: List[float] = [0.0]
    file_2 = open("INFILE", "r")
    (X,Y,) = format_10_obj.read_line(file_2.readline())
    
    file_2.close()
    file_1 = open("OUTFILE", "w")
    
    write_list = []
    write_line = format_11_obj.write_line(write_list)
    file_1.write(write_line+'\n')
    write_list = [X, Y]
    write_line = format_12_obj.write_line(write_list)
    file_1.write(write_line+'\n')
    
    file_1.close()
    return

MAIN()
