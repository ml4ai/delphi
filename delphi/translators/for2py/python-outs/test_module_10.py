import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod10 import *


def f(u: List[int], v: List[int], w: List[int], a: List[int], b: List[int], c: List[int]):
    format_10: List[str] = []
    format_10 = ['6(I5,X)']
    format_10_obj = Format(format_10)
    
    
    a[0] = (u[0] + x[0])
    b[0] = (v[0] + y[0])
    c[0] = (w[0] + z[0])
    
    write_list_stream = [x[0], y[0], z[0], a[0], b[0], c[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)

def main():
    format_10: List[str] = []
    format_10 = ['6(I5,X)']
    format_10_obj = Format(format_10)
    
    x: List[int] = [0]
    y: List[int] = [0]
    z: List[int] = [0]
    p: List[int] = [0]
    q: List[int] = [0]
    r: List[int] = [0]
    x[0] = 987
    y[0] = 876
    z[0] = 765
    f(x, y, z, p, q, r)
    
    write_list_stream = [x[0], y[0], z[0], p[0], q[0], r[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

main()
