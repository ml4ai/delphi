from typing import List
import math
from fortran_format import *

def MAIN():
    format_10 = ['F5.3']
    format_10_obj = Format(format_10)
    
    file_10 = open("infile2", "r")
    file_20 = open("outfile2", "w")
    (X,) = format_10_obj.read_line(file_10.readline())
    write_list = [X]
    write_line = format_10_obj.write_line(write_list)
    file_20.write(write_line+'\n')
    
    return

MAIN()
