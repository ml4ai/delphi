from typing import List
import math
from fortran_format import *

def MAIN():
    file_10 = open("infile1", "r")
    file_20 = open("outfile1", "w")
    read_format_10 = ['I5', 'I5']
    read_file_10 = Format(read_format_10)
    (I,J,) = read_file_10.read_line(file_10.readline())
    write_format_20 = ['I5', 'I5']
    write_file_20 = Format(write_format_20)
    write_list = [J, I]
    write_line = write_file_20.write_line(write_list)
    file_20.write(write_line)
    
    return

MAIN()
