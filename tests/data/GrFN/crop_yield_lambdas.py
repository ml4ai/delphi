import math

def update_est__assign__total_rain_1(total_rain, rain):
    return (total_rain+rain)

def update_est__condition__IF_1_0(total_rain):
    return (total_rain<=40)

def update_est__assign__yield_est_0(total_rain):
    return (-((((total_rain-40)**2)/16))+100)

def update_est__assign__yield_est_1(total_rain):
    return (-(total_rain)+140)

def update_est__decision__yield_est_2(IF_1_0, yield_est_1, yield_est_0):
    return yield_est_0 if IF_1_0 else yield_est_1

def crop_yield__assign__max_rain_1():
    return 4.0

def crop_yield__assign__consistency_1():
    return 64.0

def crop_yield__assign__absorption_1():
    return 0.6

def crop_yield__assign__yield_est_1():
    return 0

def crop_yield__assign__total_rain_1():
    return 0

def crop_yield__assign__rain_0(day, consistency, max_rain, absorption):
    return ((-((((day-16)**2)/consistency))+max_rain)*absorption)

