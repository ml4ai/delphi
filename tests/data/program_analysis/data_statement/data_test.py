import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random


def main():
    format_10: List[str] = [None]
    format_10 = ['A', 'I1', '2X', 'A', 'I1', '2X', 'A', 'I1']
    format_10_obj = Format(format_10)
    
    format_11: List[str] = [None]
    format_11 = ["''"]
    format_11_obj = Format(format_11)
    
    format_20: List[str] = [None]
    format_20 = ['A', 'A']
    format_20_obj = Format(format_20)
    
    format_25: List[str] = [None]
    format_25 = ['A']
    format_25_obj = Format(format_25)
    
    format_30: List[str] = [None]
    format_30 = ['A', 'F3.1']
    format_30_obj = Format(format_30)
    
    format_35: List[str] = [None]
    format_35 = ['F3.1']
    format_35_obj = Format(format_35)
    
    format_40: List[str] = [None]
    format_40 = ['4(F3.1,2X)']
    format_40_obj = Format(format_40)
    
    format_45: List[str] = [None]
    format_45 = ['2(F3.1,2X)']
    format_45_obj = Format(format_45)
    
    a: List[int] = [None]
    b: List[int] = [None]
    c: List[int] = [None]
    z = String(4)
    m = String(1)
    n = String(1)
    vec = Array(float, [(0, 5)])
    x: List[float] = [None]
    y: List[float] = [None]
    pair1 = Array(float, [(0, 2)])
    pair2 = Array(float, [(0, 2)])
    pair3 = Array(float, [(0, 2)])
    multi = Array(float, [(0, 5), (0, 4)])
    mult_a = Array(float, [(0, 2), (0, 2)])
    mult_b = Array(float, [(0, 2), (0, 2)])
    i: List[int] = [None]
    a[0] = 1
    b[0] = 2
    c[0] = 3
    z.set_("Overwrite")
    m.set_("A")
    n.set_("A")
    x[0] = 9.0
    vec.set_((1), 9.0)
    vec.set_((2), 9.0)
    vec.set_((3), 0.1)
    vec.set_((4), 0.5)
    vec.set_((5), 0.5)
    y[0] = 6.0
    pair1.set_((1), 4.0)
    pair1.set_((2), 2.0)
    pair2.set_((1), 4.0)
    pair3.set_((2), 2.0)
    i_iterator: List[int] = [None]
    j_iterator: List[int] = [None]
    for i_iterator[0] in range(1, 5+1):
        for j_iterator[0] in range(1, 4+1):
            multi.set_((i_iterator[0], j_iterator[0]), 2.5)
    mult_a.set_((1, 1), 1.0)
    mult_a.set_((2, 1), 2.0)
    mult_a.set_((1, 2), 3.0)
    mult_a.set_((2, 2), 4.0)
    mult_b.set_((1, 1), 1.1)
    mult_b.set_((1, 2), 2.1)
    mult_b.set_((2, 1), 3.1)
    mult_b.set_((2, 2), 4.1)
    write_list_stream = ["A: ", a[0], "B: ", b[0], "C: ", c[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["Z: ", z]
    write_line = format_20_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["M: ", m]
    write_line = format_20_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["N: ", n]
    write_line = format_20_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["X: ", x[0]]
    write_line = format_30_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["VEC: "]
    write_line = format_25_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    for i[0] in range(1, 5+1):
        write_list_stream = [vec.get_((i[0]))]
        write_line = format_35_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["Y: ", y[0]]
    write_line = format_30_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["PAIR1: "]
    write_line = format_25_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    for i[0] in range(1, 2+1):
        write_list_stream = [pair1.get_((i[0]))]
        write_line = format_35_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["PAIR2: "]
    write_line = format_25_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    for i[0] in range(1, 2+1):
        write_list_stream = [pair2.get_((i[0]))]
        write_line = format_35_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["PAIR3: "]
    write_line = format_25_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    for i[0] in range(1, 2+1):
        write_list_stream = [pair3.get_((i[0]))]
        write_line = format_35_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["MULTI: "]
    write_line = format_25_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    for i[0] in range(1, 5+1):
        write_list_stream = [multi.get_((i[0], 1)), multi.get_((i[0], 2)), multi.get_((i[0], 3)), multi.get_((i[0], 4))]
        write_line = format_40_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["MULT_A: "]
    write_line = format_25_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    for i[0] in range(1, 2+1):
        write_list_stream = [mult_a.get_((i[0], 1)), mult_a.get_((i[0], 2))]
        write_line = format_45_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = ["MULT_B: "]
    write_line = format_25_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    for i[0] in range(1, 2+1):
        write_list_stream = [mult_b.get_((i[0], 1)), mult_b.get_((i[0], 2))]
        write_line = format_45_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    
    
    
    
    
    
    
    

main()
