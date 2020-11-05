from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import numpy as np
import delphi.translators.for2py.math_ext as math

def SIR_simple__sir__assign__infected__0(beta: Real, s: Real, i: Real, r: Real, dt: Real):
    return (-((((beta*s)*i)/((s+i)+r)))*dt)

def SIR_simple__sir__assign__recovered__0(gamma: Real, i: Real, dt: Real):
    return ((gamma*i)*dt)

def SIR_simple__sir__assign__s__0(s: Real, infected: Real):
    return (s-infected)

def SIR_simple__sir__assign__i__0(i: Real, infected: Real, recovered: Real):
    return ((i+infected)-recovered)

def SIR_simple__sir__assign__r__0(r: Real, recovered: Real):
    return (r+recovered)

