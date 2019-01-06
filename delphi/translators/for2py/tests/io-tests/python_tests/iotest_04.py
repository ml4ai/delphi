from typing import List
import math
from fortran_format import *

def MAIN():
    file_10 = open("infile2", "r")
    file_20 = open("outfile2", "w")
    read_format_10 = ['F5.2', 'X', 'F5.2']
    read_file_10 = Format(read_format_10)
    (X,Y,) = read_file_10.read_line(file_10.readline())
    write_format_20 = ['F5.2', 'X', 'F5.2']
    write_file_20 = Format(write_format_20)
    write_list = [Y, X]
    write_line = write_file_20.write_line(write_list)
    file_20.write(write_line)
    
    return

MAIN()
