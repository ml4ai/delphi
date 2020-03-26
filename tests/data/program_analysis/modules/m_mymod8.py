import sys
import os
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from delphi.translators.for2py import intrinsics
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random


x: List[int] = [1234]

def myadd(y: List[int]):
    myadd_return: List[int] = [None]
    myadd_return[0] = (x[0] + y[0])
    return myadd_return[0]
