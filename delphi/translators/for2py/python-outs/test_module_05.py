import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod6 import *


def pgm():
    format_10: List[str] = []
    format_10 = ['I8']
    format_10_obj = Format(format_10)
    
    
    v: List[int] = [0]
    myadd(2345, v)
    write_list_stream = [v[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    
    return

pgm()
