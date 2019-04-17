import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    a = Array(int, [(1, 3), (1, 5)])
    b = Array(int, [(1, 5), (1, 3)])
    i: List[int] = [0]
    j: List[int] = [0]
    for i[0] in range(1, 3+1):
        for j[0] in range(1, 5+1):
            a.set_((i[0], j[0]), ((i[0] * j[0]) + (i[0] + j[0])))
    for i[0] in range(1, 3+1):
        for j[0] in range(1, 5+1):
            b.set_((j[0], i[0]), a.get_((i[0], 1)))
    return

main()
