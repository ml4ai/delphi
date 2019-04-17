import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod11b import *


z: List[float] = [12.345]

def print_mymod11c():
    format_11: List[str] = []
    format_11 = ["'mymod11C: X = '", 'I3', "'; Y = '", 'I3', "'; Z = '", 'F8.3', "'; U = '", 'I3', "'; V = '", 'I3']
    format_11_obj = Format(format_11)
    
    print_mymod11b()
    
    write_list_stream = [x[0], _y[0], z[0], u[0], v[0]]
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)