import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def nested_loop():
    month: List[int] = [0]
    day: List[int] = [0]
    month[0] = 1
    while (month[0] <= 12):
        day[0] = 1
        while (day[0] <= 7):
            print("Month: ", month, " DAY: ", day)
            day[0] = (day[0] + 1)
        month[0] = (month[0] + 1)

nested_loop()
