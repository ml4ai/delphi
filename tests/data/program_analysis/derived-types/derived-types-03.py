import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from random import random
@dataclass

class mytype_1:
    def __init__(self):
        self.a : int
        self.b : float

class mytype_2:
    def __init__(self):
        self.a : int
        self.b : float


def main():
    format_10: List[str] = [None]
    format_10 = ['I5', '3X', 'F7.3']
    format_10_obj = Format(format_10)
    
    x =  mytype_1()
    y =  mytype_2()
    x.a = 123
    x.b = 4.56
    y.a = (x.a * 2)
    y.b = (x.b * 3)
    
    write_list_stream = [x.a, x.b]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = [y.a, y.b]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

main()
