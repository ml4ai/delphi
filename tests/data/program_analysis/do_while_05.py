import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real


def do_while():
    month: List[int] = [None]
    month[0] = 1
    while (month[0] <= 13):
        if (month[0] == 13):
            return
        print("Month: ", month)
        month[0] = (month[0] + 1)

do_while()
