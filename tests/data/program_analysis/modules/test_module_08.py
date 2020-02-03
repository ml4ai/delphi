import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random
from delphi.translators.for2py.tmp.m_mymod8 import myadd


def pgm():
    format_10: List[str] = [None]
    format_10 = ['I8', '2X', 'I8']
    format_10_obj = Format(format_10)
    
    
    x: List[int] = [None]
    v: List[int] = [None]
    x[0] = 5678
    v[0] = myadd(x)
    
    write_list_stream = [x[0], v[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

pgm()
