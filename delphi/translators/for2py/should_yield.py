import sys
from typing import List
import math
from fortran_format import *
from for2py_arrays import *


def crop_yield():
    day: List[int] = [0]
    
    rain: List[float] = [0.0]
    
    yield_est: List[float] = [0.0]
    
    total_rain: List[float] = [0.0]
    
    max_rain: List[float] = [0.0]
    
    consistency: List[float] = [0.0]
    
    absorption: List[float] = [0.0]
    
    max_rain[0] = 4.0
    consistency[0] = 64.0
    absorption[0] = 0.6
    yield_est[0] = 50.5
    total_rain[0] = 0
    print(["Day "], [max_rain[0]], [" Estimate: "], [consistency[0]])
    print(["Crop Yield(%): "], [yield_est[0]])

crop_yield()
