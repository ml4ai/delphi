import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


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
                    print("MONTH: ", month, " DAY: ", day, ", SUNDAY")
                else:
                    if (day[0] == 2):
                        print("MONTH: ", month, " DAY: ", day, ", MONDAY")
                    else:
                        if (day[0] == 3):
                            print("MONTH: ", month, " DAY: ", day, ", TUESDAY")
                        else:
                            if (day[0] == 4):
                                print("MONTH: ", month, " DAY: ", day, ", WEDNESDAY")
                            else:
                                if (day[0] == 5):
                                    print("MONTH: ", month, " DAY: ", day, ", THURSDAY")
                                else:
                                    if (day[0] == 6):
                                        print("MONTH: ", month, " DAY: ", day, ", FRIDAY")
                                    else:
                                        print("MONTH: ", month, " DAY: ", day, ", SATURDAY")
                day[0] = (day[0] + 1)
            date[0] = (date[0] + 1)
        month[0] = (month[0] + 1)

triple_nested()
