import numpy as np
from itertools import product


### Function whose interval has to be computed
def PETPT(x1, x2):

    y = x2*1.1

    if x1 > 35.0:
        y = x2*((x1-35.0)*0.05 + 1.1)

    return y



size = 10

x1 = np.linspace(16.1, 36.7, size)  # interval range for x1
x2 = np.linspace(0.048, 8.219, size)  # interval range for x2

# Calculate all possible values of PETPT in the two dimensional space spanned by x1 and
# x2 
y1 = [PETPT(*args) for args in product(x1, x2)]

#Upper and Lower Bounds of the interval
print("Lower Bound", min(y1))
print("Upper Bound", max(y1))





