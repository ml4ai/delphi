import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod3 import *
from m_mymod4 import *
from m_mymod5 import *


def pgm():
    format_10: List[str] = []
    format_10 = ['I7', 'I5', 'F8.4']
    format_10_obj = Format(format_10)
    
    
    
    
    
    write_list_stream = [x[0], y[0], pi[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

pgm()
