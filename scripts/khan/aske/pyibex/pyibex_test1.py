from pyibex import *

f1 = Function("x", "y", "x*sin(x+y) - 2")

f2 = Function("x[2]", "x[0]*sin(x[0] + x[1]) - 2")

f3 = Function("x1", "x1*1.1")

x1 = Interval(0, 1)
x2 = Interval(2, 5)

f4 = Function("x1", "x2", "f3")

x = IntervalVector([[1, 3], [1,4]])

print(f1.eval(x))

print(f2.eval(x))
