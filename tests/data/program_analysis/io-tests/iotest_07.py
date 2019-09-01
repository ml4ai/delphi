import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def m():
    format_10: List[str] = [None]
    format_10 = ['I2', 'I3']
    format_10_obj = Format(format_10)
    
    z: List[int] = [None]
    a: List[int] = [None]
    z[0] = 12
    a[0] = 567
    write_list_stream = [z[0], a[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    
    return

m()
