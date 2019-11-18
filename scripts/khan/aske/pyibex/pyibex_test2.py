from pyibex import *
import math

f = Function("myfunction.txt")
X0 = IntervalVector([[16.1,36.7], [0.0,23.9], [0.0, 1.0], [0.0, 4.77], [2.45, 27.8]])
print(f.eval(X0))

