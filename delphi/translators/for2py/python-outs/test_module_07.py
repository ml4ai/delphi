import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod7 import *


def pgm():
    format_10: List[str] = []
    format_10 = ['I8', '2X', 'I8']
    format_10_obj = Format(format_10)
    
    
    x: List[int] = [0]
    v: List[int] = [0]
    x[0] = 5678
    myadd(x, v)
    
    write_list_stream = [x[0], v[0]]
    output_fmt = list_output_formats(["integer","integer",])
    write_stream_obj = Format(output_fmt)
    write_line = write_stream_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

pgm()
