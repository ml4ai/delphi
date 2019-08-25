#!/usr/bin/env python3

"""
File: 
    static_save.py
Purpose: 
    A decorator function to initialize a list of static variables to a None.

Usage:
        In the python file, add the following line above the function
        definition which is to be decorated.

        @static_vars([<variable_list])
        def target_function():

        where,
            <variable_list> = List of static variables within the function
            "target_function"
"""

from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def static_vars(var_list):
    # This code is part of the runtime system
    def decorate(func):
        for var in var_list:
            setattr(func, var["name"], var["call"])
        return func
    return decorate
