""" This module implements several arithmetic and logical operations such that it
maintains single-precision across variables and operations. It overloads
double-precision floating variables into single-precision using numpy's
float32 method, forcing them to stay in single-precision type. This avoids a
possible butterfly effect in numerical operations involving large floating
point calculations.

Usage:
    To convert a float variable into Float32 type, do the following
    target = Float32(float_variable)
    OR
    target = Float32(5.0)
    where <target> is now a Float32 object.

Now, any arithmetic/logical operation involving <float_variable> will involve
single-point calculations.

Example Usage:
    eps = Float32(1.0)
    while eps + 1.0 > 1.0:
    ____eps /= 2.0
    eps *= 2.0

Authors:
    Saumya Debray
    Pratik Bhandari

"""
from numbers import Real, Number
import math

from numpy import float32


class Float32(Real):
    """ This class converts float variables into float32 type for single-precision
    calculation and overloads the default arithmetic and logical operations.
    All methods below follow a similar
    input/output type.

    Input: Either a Float32 variable or a variable of a supported type.
    Output: A Float32 object of the result of the operation.
    """

    def __init__(self, val):
        """ This constructor converts the float variable 'val' into numpy's
        float32 type.
        """
        self._val = float32(val)

    def __add__(self, other):
        """ Addition
        """
        return Float32(self._val+self.__value(other))

    def __radd__(self, other):
        """ Reverse Addition
        """
        return Float32(self._val+self.__value(other))

    def __truediv__(self, other):
        """ True Division
        """
        return Float32(self._val/self.__value(other))

    def __rtruediv__(self, other):
        """ Reverse True Division
        """
        return Float32(self.__value(other)/self._val)

    def __floordiv__(self, other):
        """ Floor Division
        """
        return self._val//self.__value(other)

    def __rfloordiv__(self, other):
        """ Reverse Floor Division
        """
        return self.__value(other)//self._val

    def __mul__(self, other):
        """ Multiplication
        """
        return Float32(self._val*self.__value(other))

    def __rmul__(self, other):
        """ Reverse sMultiplication
        """
        return Float32(self._val*self.__value(other))

    def __sub__(self, other):
        """ Subtraction
        """
        return Float32(self._val-self.__value(other))

    def __rsub__(self, other):
        """ Reverse Subtraction
        """
        return Float32(self.__value(other)-self._val)

    def __eq__(self, other):
        """ Logical Equality
        """
        return self._val == self.__value(other)

    def __gt__(self, other):
        """ Logical greater than
        """
        return self._val > self.__value(other)

    def __ge__(self, other):
        """ Logical greater than or equals to
        """
        return self._val >= self.__value(other)

    def __lt__(self, other):
        """ Logical less than
        """
        return self._val < self.__value(other)

    def __le__(self, other):
        """ Logical less than or equals to
        """
        return self._val <= self.__value(other)

    def __ne__(self, other):
        """ Logical not equal to
        """
        return self._val != self.__value(other)

    def __str__(self):
        """ Float-to-string conversion
        """
        return str(self._val)

    def __abs__(self):
        """ Absolute value conversion
        """
        return abs(self._val)

    def __neg__(self):
        """ Value negation
        """
        return -self._val

    def __mod__(self, other):
        """ Modulo operation
        """
        return Float32(self._val % self.__value(other))

    def __rmod__(self, other):
        """ Reverse Modulo operation
        """
        return Float32(self.__value(other) % self._val)

    def __and__(self, other):
        """ Bitwise 'and' of two variables
        """
        return Float32(self._val and self.__value(other))

    def __or__(self, other):
        """ Bitwise 'or' of two variables
        """
        return Float32(self._val or self.__value(other))

    def __pos__(self):
        """ Return positive value
        """
        return +self._val

    def __pow__(self, other):
        """ Power operation
        """
        return Float32(self._val ** self.__value(other))

    def __rpow__(self, other):
        """ Reverse Power operation
        """
        return Float32(self.__value(other) ** self._val)

    # ==========================================================================
    # Added by Paul to extend the Real class
    def __float__(self):
        return float(self._val)

    def __round__(self, ndigits=None):
        return round(self._val, ndigits=ndigits)

    def __ceil__(self):
        return Float32(math.ceil(self._val))

    def __floor__(self):
        return Float32(math.floor(self._val))

    def __trunc__(self):
        return int(self._val)
    # ==========================================================================

    def __value(self, other):
        """ This method checks whether the variable is a Float32 type or not and
        returns it's value likewise.
        """
        if isinstance(other, Float32):
            return other._val
        elif isinstance(other, Number):
            return other
        else:
            raise TypeError(f"Unusable type ({type(other)}) with Float32")
