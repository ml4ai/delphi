import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 100)
y = -x**2 + 0.5*x**4

plt.figure()
plt.plot(x, y, c='b')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axvline(x=1, c='black', linestyle = '--' )
plt.axvline(x=-1, c='black', linestyle = '--' )
plt.show()


