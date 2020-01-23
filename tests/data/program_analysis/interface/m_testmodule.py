import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random

@dataclass
class ControlType:
    def __init__(self):
        self.mesic : str
        self.crop : str
        self.model : str
        self.filex : str
        self.fileio : str
        self.dssatp : str
        self.das : int
        self.nyrs : int
        self.yrdif : int

@dataclass
class SwitchType:
    def __init__(self):
        self.fname : str
        self.idetc : str
        self.ideto : str
        self.ihari : str
        self.iswche : str
        self.iswpho : str
        self.meevp : str
        self.mesom : str
        self.metmp : str
        self.iferi : str
        self.nswi : int

@dataclass
class TransferType:
    def __init__(self):
        self.control : controltype
        self.iswitch : switchtype
        self.output : outputtype
        self.plant : planttype
        self.mgmt : mgmttype
        self.nitr : nitype
        self.orgc : orgctype
        self.soilprop : soiltype
        self.spam : spamtype
        self.water : wattype
        self.weather : weathtype
        self.pdlabeta : pdlabetatype


maxfiles: List[int] = [500]
save_data =  transfertype()
def get (arg1=None):
    num_passed_args = 0
    if arg1 != None:
        num_passed_args += 1

    if num_passed_args == 1:
        if isinstance(arg1[0], ControlType):
            get_control(arg1)
        if isinstance(arg1[0], SwitchType):
            get_iswitch(arg1)

def put (arg1=None):
    num_passed_args = 0
    if arg1 != None:
        num_passed_args += 1

    if num_passed_args == 1:
        if isinstance(arg1[0], SwitchType):
            put_iswitch(arg1)


def get_control(control_arg: List[controltype]):
    control_arg[0] = save_data.control
    

def put_control(control_arg: List[controltype]):
    save_data.control = control_arg[0]
    

def get_iswitch(iswitch_arg: List[switchtype]):
    iswitch_arg[0] = save_data.iswitch
    

def put_iswitch(iswitch_arg: List[switchtype]):
    save_data.iswitch = iswitch_arg[0]
    