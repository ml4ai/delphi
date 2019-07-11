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


class ConstraintSetter(ABCMeta):
    def __init__(self, d: Interval, r: Interval) -> None:
        self.__domain = d
        self.__range = r

    def is_domain_left_inclusive(self) -> bool:
        return self.__domain.left_inclusive

    def is_domain_right_inclusive(self) -> bool:
        return self.__domain.right_inclusive

    def is_range_left_inclusive(self) -> bool:
        return self.__range.left_inclusive

    def is_range_right_inclusive(self) -> bool:
        return self.__range.right_inclusive


class ConstraintShifter(ABCMeta):
    def __init__(self, d_scale: Callable, r_scale: Callable) -> None:
        return NotImplemented


class ConstraintScalar(ABCMeta):
    def __init__(self, d_scale: Callable, r_scale: Callable) -> None:
        return NotImplemented
# ==============================================================================


# ==============================================================================
# ELEMENTARY MATH OPERATIONS
# ==============================================================================
class AddOpScalar(ConstraintScalar):
    def __init__(self, amt) -> None:
        super().__init__(None, amt)

# ==============================================================================


# ==============================================================================
# TRIGONOMETRIC FUNCTION CONSTRAINTS
# ==============================================================================
class CosConstraint(ConstraintSetter):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-1, 1, True, True)
        super().__init__(d, r)


class SinConstraint(ConstraintSetter):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-1, 1, True, True)
        super().__init__(d, r)


class TanConstraint(ConstraintSetter):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(-math.inf, math.inf, False, False)
        super().__init__(d, r)


class ACosConstraint(ConstraintSetter):
    def __init__(self) -> None:
        d = Interval(-1, 1, True, True)
        r = Interval(0, math.pi, True, True)
        super().__init__(d, r)


class ASinConstraint(ConstraintSetter):
    def __init__(self) -> None:
        d = Interval(-1, 1, True, True)
        r = Interval(-math.pi/2, math.pi/2, True, True)
        super().__init__(d, r)


class ATanConstraint(ConstraintSetter):
    def __init__(self) -> None:
        d = Interval(-math.inf, math.inf, False, False)
        r = Interval(0, math.pi, False, False)
        super().__init__(d, r)
# ==============================================================================
