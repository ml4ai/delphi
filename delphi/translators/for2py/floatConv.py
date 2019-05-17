from numpy import float32

################################################################################
class FloatConv:
    def __init__(self, val):
        self._val = float32(val)

    def __add__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return FloatConv(self._val+n)

    def __radd__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return FloatConv(self._val+n)

    def __truediv__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return FloatConv(self._val/n)

    def __rtruediv__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return FloatConv(n/self._val)

    def __mul__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return FloatConv(self._val*n)

    def __sub__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return FloatConv(self._val-n)

    def __rsub__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return FloatConv(n-self._val)

    def __eq__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return self._val == n

    def __gt__(self, other):
        if isinstance(other, FloatConv):
            n = other._val
        else:
            n = other

        return self._val > n

    def __str__(self):
        return str(self._val)

################################################################################

def eps_native():
    eps = 1.0
    while (eps + 1.0 > 1.0):
        eps = eps/2.0
    eps = eps*2.0

    return eps

def eps_numpy_32():
    eps = float32(1.0)
    while (eps + 1.0 > 1.0):
        eps = eps/2.0
    eps = eps*2.0

    return eps

def eps_numpy_32_packaged_0():
    eps = float32(1.0)
    while (float32(eps + 1.0) > 1.0):
        eps = float32(eps/2.0)
    eps = float32(eps*2.0)

    return eps

def eps_numpy_32_packaged_1():
    eps = float32(1.0)
    while (eps + 1.0 > 1.0):
        eps = float32(eps/2.0)
    eps = float32(eps*2.0)

    return eps

def eps_FloatConv():
    eps = FloatConv(1.0)
    while (eps + 1.0 > 1.0):
        eps = eps/2.0
    eps = eps * 2.0

    return eps

def eps_FloatConv_1():
    eps = FloatConv(1.0)
    while (eps + 1.0 > 1.0):
        eps /= 2.0
    eps *= 2.0

    return eps


def f(x, y):
    x[0] = y[0]+1

def main():
    print("NATIVE   : {}".format(str(eps_native())))
    print("Numpy_32 : {}".format(str(eps_numpy_32())))
    print("Numpy_32_Packaged_0 : {}".format(str(eps_numpy_32_packaged_0())))
    print("Numpy_32_Packaged_1 : {}".format(str(eps_numpy_32_packaged_1())))
    print("FloatConv   : {}".format(str(eps_FloatConv())))
    print("FloatConv_1 : {}".format(str(eps_FloatConv_1())))

    a = [FloatConv(1.0)]
    b = [FloatConv(2.0)]
    print(a[0])
    f(a, b)
    print(a[0])