from abc import ABCMeta, abstractmethod
import math
from typing import Callable


# ==============================================================================
# ABSTRACT AND SUPPORTING CLASSES
# ==============================================================================
class Interval:
    def __init__(self, l: float, r: float, l_inc: bool, r_inc: bool) -> None:
        self.left_bound = l
        self.right_bound = r
        self.left_inclusive = l_inc
        self.right_inclusive = r_inc


class Constraint(ABCMeta):
    def __init__(self, d: Interval, r: Interval, f: Callable) -> None:
        self.__domain = d
        self.__range = r
        self.__inverse = f

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self, constraint_type):
        return f"{constraint_type}\n" + \
            f"\tD: {self.__domain}\n" + \
            f"\tR: {self.__range}\n" + \
            f"\tI: {self.__inverse}"

    def is_domain_left_inclusive(self) -> bool:
        return self.__domain.left_inclusive

    def is_domain_right_inclusive(self) -> bool:
        return self.__domain.right_inclusive

    def is_range_left_inclusive(self) -> bool:
        return self.__range.left_inclusive

    def is_range_right_inclusive(self) -> bool:
        return self.__range.right_inclusive
# ==============================================================================


# ==============================================================================
# ELEMENTARY MATH OPERATIONS
# ==============================================================================
class AddOpConstraint(Constraint):
    def __init__(self, operand) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-math.inf, math.inf, False, False)
        inv = lambda x: x - operand
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class SubOpConstraint(Constraint):
    def __init__(self, operand) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-math.inf, math.inf, False, False)
        inv = lambda x: x + operand
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class DivOpConstraint(Constraint):
    def __init__(self, operand) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-math.inf, math.inf, False, False)
        inv = lambda x: x * operand
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class MulOpConstraint(Constraint):
    def __init__(self, operand) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-math.inf, math.inf, False, False)
        inv = lambda x: x / operand
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)
# ==============================================================================


# ==============================================================================
# COMMON FUNCTION OPERATIONS
# ==============================================================================
class ExpConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(0, math.inf, False, True)
        inv = lambda x: math.log(x)
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class NLogConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(0, math.inf, False, False)
        r = Interval(-math.inf, math.inf, False, True)
        inv = lambda x: math.exp(x)
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)
# ==============================================================================


# ==============================================================================
# TRIGONOMETRIC FUNCTION CONSTRAINTS
# ==============================================================================
class CosConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-1, 1, True, True)
        inv = lambda x: math.acos(x)        # NOTE: This value will be in radians
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class SinConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-1, 1, True, True)
        inv = lambda x: math.asin(x)
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class TanConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-math.inf, math.inf, False, False)
        inv = lambda x: math.atan(x)
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class ACosConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(-1, 1, True, True)
        r = Interval(0, math.pi, True, True)
        inv = lambda x: math.cos(x)
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class ASinConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(-1, 1, True, True)
        r = Interval(-math.pi/2, math.pi/2, True, True)
        inv = lambda x: math.sin(x)
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)


class ATanConstraint(Constraint):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(0, math.pi, False, False)
        inv = lambda x: math.tan(x)
        super().__init__(d, r, inv)

    def __str__(self):
        return super().__str__(self.__name__)
# ==============================================================================
