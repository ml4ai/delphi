from numpy import float32


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
