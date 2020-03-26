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


def triple_nested():
    month: List[int] = [None]
    date: List[int] = [None]
    day: List[int] = [None]
    month[0] = 1
    while (month[0] <= 12):
        date[0] = 1
        while (date[0] <= 7):
            day[0] = 1
            while (day[0] <= date[0]):
                if (day[0] == 1):
                    print("month: ", month, " day: ", day, ", sunday")
                else:
                    if (day[0] == 2):
                        print("month: ", month, " day: ", day, ", monday")
                    else:
                        if (day[0] == 3):
                            print("month: ", month, " day: ", day, ", tuesday")
                        else:
                            if (day[0] == 4):
                                print("month: ", month, " day: ", day, ", wednesday")
                            else:
                                if (day[0] == 5):
                                    print("month: ", month, " day: ", day, ", thursday")
                                else:
                                    if (day[0] == 6):
                                        print("month: ", month, " day: ", day, ", friday")
                                    else:
                                        print("month: ", month, " day: ", day, ", saturday")
                day[0] = (day[0] + 1)
            date[0] = (date[0] + 1)
        month[0] = (month[0] + 1)

triple_nested()
