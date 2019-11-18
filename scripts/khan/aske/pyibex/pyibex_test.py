# Load pyIbex lib
from pyibex import *

# Create new Intervals
a = Interval.EMPTY_SET
a = Interval.ALL_REALS
a  = Interval(-2, 3)

# Create IntervalVector
b = IntervalVector( 2, a)
c = IntervalVector([1,2,3])
d = IntervalVector([[-1,3], [3,10], [-3, -1]])

e = IntervalVector( (a, Interval(-1,0), Interval(0)) )

print(e)

# Operations
e = c & d
print(e)
e = c+d
print(e)
e = a * c
print(e)

