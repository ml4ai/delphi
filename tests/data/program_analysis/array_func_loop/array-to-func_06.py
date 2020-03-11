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


def update_mean_var(means: Array, vars: Array, k: List[int], n: List[int], runs: List[int]):
    tmax: List[int] = [100]
    
    
    means.set_((k[0]), (means.get_((k[0])) + ((n[0] - means.get_((k[0]))) / (runs[0] + 1))))
    vars.set_((k[0]), (vars.get_((k[0])) + (((runs[0] / (runs[0] + 1)) * (n[0] - means.get_((k[0])))) * (n[0] - means.get_((k[0]))))))
    

def gillespie(means: Array, vars: Array):
    tmax: List[int] = [100]
    runs: List[int] = [None]
    sample: List[int] = [None]
    i: List[int] = [None]
    j: List[int] = [None]
    
    
    samples = Array(float, [(0, tmax[0])])
    for i[0] in range(0, tmax[0]+1):
        samples.set_((i[0]), i[0])
    for runs[0] in range(0, tmax[0]+1):
        j[0] = 0
        while (j[0] <= tmax[0]):
            update_mean_var(means, vars, j, sample, runs)
            j[0] = (j[0] + 1)
        sample[0] = samples.get_((runs[0]))

def main():
    tmax: List[int] = [100]
    means = Array(float, [(0, tmax[0])])
    vars = Array(float, [(0, tmax[0])])
    gillespie(means, vars)

main()
