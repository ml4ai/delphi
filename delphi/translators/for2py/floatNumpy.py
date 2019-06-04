from numpy import float32

################################################################################


class Float32:
    def __init__(self, val):
        self._val = float32(val)

    def __add__(self, other):
        return Float32(self._val+self.value(other))

    def __radd__(self, other):
        return Float32(self._val+self.value(other))

    def __truediv__(self, other):
        return Float32(self._val/self.value(other))

    def __rtruediv__(self, other):
        return Float32(self.value(other)/self._val)

    def __mul__(self, other):
        return Float32(self._val*self.value(other))

    def __sub__(self, other):
        return Float32(self._val-self.value(other))

    def __rsub__(self, other):
        return Float32(self.value(other)-self._val)

    def __eq__(self, other):
        return self._val == self.value(other)

    def __gt__(self, other):
        return self._val > self.value(other)

    def __str__(self):
        return str(self._val)

    def value(self, other):
        if isinstance(other, Float32):
            return other._val
        else:
            return other

################################################################################


def eps_native():
    eps = 1.0
    while eps + 1.0 > 1.0:
        eps = eps/2.0
    eps = eps*2.0

    return eps


def eps_numpy_32():
    eps = float32(1.0)
    while eps + 1.0 > 1.0:
        eps = eps/2.0
    eps = eps*2.0

    return eps


def eps_numpy_32_packaged_0():
    eps = float32(1.0)
    while float32(eps + 1.0) > 1.0:
        eps = float32(eps/2.0)
    eps = float32(eps*2.0)

    return eps


def eps_numpy_32_packaged_1():
    eps = float32(1.0)
    while eps + 1.0 > 1.0:
        eps = float32(eps/2.0)
    eps = float32(eps*2.0)

    return eps


def eps_float32():
    eps = Float32(1.0)
    while eps + 1.0 > 1.0:
        eps = eps/2.0
    eps = eps * 2.0

    return eps


def eps_float32_1():
    eps = Float32(1.0)
    while eps + 1.0 > 1.0:
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
    print("Float32   : {}".format(str(eps_float32())))
    print("Float32_1 : {}".format(str(eps_float32_1())))

    a = [Float32(1.0)]
    b = [Float32(2.0)]
    print(a[0])
    f(a, b)
    print(a[0])
