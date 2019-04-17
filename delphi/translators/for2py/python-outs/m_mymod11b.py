import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod11a import *


u: List[int] = [567]
v: List[int] = [678]

def print_mymod11b():
    format_10: List[str] = []
    format_10 = ["'mymod11B: X = '", 'I3', "'; Y = '", 'I3', "'; Z = '", 'I3', "'; U = '", 'I3', "'; V = '", 'I3']
    format_10_obj = Format(format_10)
    
    
    write_list_stream = [x[0], y[0], z[0], u[0], v[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)