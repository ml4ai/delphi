#from pyibex import IntervalVector, Function
from pyibex import *
from math import *

def petpt(tmax, tmin, msalb, xhlai, srad):

#Variable tmax

    td = 0.6*tmax + 0.4*tmin

    if xhlai < 0.0:
        albedo = msalb
    else:
        albedo = 0.23 - (0.23 - msalb)*exp(-0.75*xhlai)

    slang = srad*23.923
    eeq = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo)*(td+29.0)
    eo = eeq*1.1

    if tmax > 35.0:
        eo = eeq*((tmax-35.0)*0.05 + 1.1)
    elif tmax < 5.0:
        eo = eeq*0.01*exp(0.18*(tmax+20.0))

    eo = max(eo, 0.0001)

    return eo

#X0 = IntervalVector(5, [-1,1])
X0 = IntervalVector([[16.1,36.7], [0.0,23.9], [0.0, 1.0], [0.0, 4.77], [2.45, 27.8]])
# x1 = Interval(16.1,36.7)
# x2 = Interval(0.0, 23.9)
# x3 = Interval(0.0, 1.0)
# x4 = Interval(0.0, 4.77)
# x5 = Interval(2.45, 27.8)

f = Function("x1", "x2", "x3", "x4", "x5", "")
#petpt(x1, x2, x3, x4, x5)


f = Function("x", "y", "x*sin(x+y) -2")
#X0 = IntervalVector([[1,3], [1,4]])
X0 = IntervalVector(2, [1,4])
f.eval(X0)

f = Function("x1", "x2", "x3", "x4", "x5", "x1+x2+x3+x4+x5")
X0 = IntervalVector(5, [1,4])
f.eval(X0)
