import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def factorial():
    format_10: List[str] = [None]
    format_10 = ["'i = '", 'I3', "'; fact = '", 'I8']
    format_10_obj = Format(format_10)
    
    i: List[int] = [None]
    n: List[int] = [None]
    fact: List[int] = [None]
    goto_flag_1: List[bool] = [None]
    goto_flag_1[0] = True
    label_flag_3: List[bool] = [None]
    label_flag_3[0] = True
    while label_flag_3[0]:
        if not(goto_flag_1[0]):
            label_flag_2: List[bool] = [None]
            label_flag_2[0] = True
            while label_flag_2[0]:
                i[0] = (i[0] + 1)
                fact[0] = (fact[0] * i[0])
                write_list_stream = [i[0], fact[0]]
                write_line = format_10_obj.write_line(write_list_stream)
                sys.stdout.write(write_line)
                if (i[0] == n[0]):
                    return
                label_flag_2[0] = True
        if goto_flag_1[0]:
            n[0] = 10
            fact[0] = 1
            i[0] = 0
            goto_flag_1[0] = False
        label_flag_3[0] = True
    

factorial()
