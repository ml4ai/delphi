import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass

@dataclass
class mytype:
    k: int = None
    v: float = None


def main():
    format_10: List[str] = []
    format_10 = ['2(I5,X,F6.3)']
    format_10_obj = Format(format_10)
    
    x =  mytype()
    y =  mytype()
    x.k = 12
    x.v = 3.456
    y.k = 21
    y.v = 4.567
    
    write_list_stream = [x.k, y.v, y.k, x.v]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

main()
