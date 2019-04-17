import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass



def foo_int(x: List[int], result: List[int]):    result[0] = ((47 * x[0]) + 23)    
def foo_real(x: List[float], result: List[int]):    result[0] = ((int(x) * 31) + 17)    
def foo_bool(x: List[bool], result: List[int]):    if x[0]:        result[0] = 937    else:        result[0] = -(732)    