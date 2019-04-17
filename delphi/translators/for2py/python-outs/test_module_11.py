import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod11c import *


def pgm():
    format_10: List[str] = []
    format_10 = ["'pgm main: X = '", 'I3', "'; Y = '", 'F7.3', "'; Z = '", 'F7.3', "'; U = '", 'F7.3', "'; V = '", 'I3']
    format_10_obj = Format(format_10)
    
    
    y: List[float] = [98.765]
    u: List[float] = [87.654]
    print_mymod11c()
    
    write_list_stream = [x[0], y[0], z[0], u[0], v[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

pgm()
