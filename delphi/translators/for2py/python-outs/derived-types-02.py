import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass

@dataclass
class mytype:
    i: int = None
    a = Array(float, [(1, 3)])


def main():
    format_10: List[str] = []
    format_10 = ['4(I3,3X,F5.3)']
    format_10_obj = Format(format_10)
    
    i: List[int] = [0]
    j: List[int] = [0]
    var =  mytype()
    x = Array(mytype(), [(1, 3)])
    for z in range(1, 3+1):
        obj = mytype()
        x.set_(z, obj)
    
    var.i = 3
    for i[0] in range(1, var.i+1):
        var.a.set_((i[0]), (var.i + i[0]))
        for j[0] in range(1, var.i+1):
            x.i).a.set_((j[0]), ((i[0] + j[0]) / 2.0))
    
    for i[0] in range(1, var.i+1):
        write_list_stream = [i[0], var.a.get_((i[0])), x.get_((i[0])).a.get_((1)), x.get_((i[0])).a.get_((2)), x.get_((i[0])).a.get_((3))]
        write_line = format_10_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    return

main()
