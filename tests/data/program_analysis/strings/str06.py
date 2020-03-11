import sys
import os
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


def main():
    format_10: List[str] = [None]
    format_10 = ['A', '"; "', 'A']
    format_10_obj = Format(format_10)
    
    str1 = String(10)
    str2 = String(5)
    str1.set_("abcdefgh")
    str2.set_(str1.get_substr(3, 8))
    str1.set_substr(2, 4, str2)
    write_list_stream = [str1, str2]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    
    return

main()
