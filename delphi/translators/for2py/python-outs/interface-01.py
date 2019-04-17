import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod import *


def main():
    format_10: List[str] = []
    format_10 = ['I5']
    format_10_obj = Format(format_10)
    
    
    x: List[int] = [0]
    
    foo(12, x)
    write_list_stream = [x[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    foo(12.0, x)
    write_list_stream = [x[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    foo(true, x)
    write_list_stream = [x[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

main()
