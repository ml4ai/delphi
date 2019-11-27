from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import delphi.translators.for2py.math_ext as math

def SIR_Gillespie_SD_multi_module__update_mean_var__assign__tmax__0():
    return 100

def SIR_Gillespie_SD_multi_module__update_mean_var__assign__means_k__0(k: int, means, n: int, runs: int):
    means[k] = (means[k]+((n-means[k])/(runs+1)))
    return means[k]

def SIR_Gillespie_SD_multi_module__update_mean_var__assign__vars_k__0(k: int, vars, runs: int, n: int, means):
    vars[k] = (vars[k]+(((runs/(runs+1))*(n-means[k]))*(n-means[k])))
    return vars[k]

