import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def m():
    x: List[float] = [None]
    y: List[float] = [None]
    z: List[int] = [None]
    x[0] = 123.456
    z[0] = 12
    y[0] = (x[0] * z[0])
    write_list_stream = [x[0], y[0], z[0]]
    output_fmt = list_output_formats(["REAL","REAL","INTEGER",])
    write_stream_obj = Format(output_fmt)
    write_line = write_stream_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

m()
