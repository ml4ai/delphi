import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod1 import *


def pgm():
    format_10: List[str] = []
    format_10 = ['I5']
    format_10_obj = Format(format_10)
    
    
    
    write_list_stream = [x[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

pgm()
