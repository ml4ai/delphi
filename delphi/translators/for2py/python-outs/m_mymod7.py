import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


x: List[int] = [1234]

def myadd(y: List[int], sum: List[int]):
    sum[0] = (x[0] + y[0])