import sys
import os
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
        self.mesic = String(1)
        self.rnmode = String(1)
        self.crop = String(2)
        self.model = String(8)
        self.ename = String(8)
        self.filex = String(12)
        self.fileio = String(30)
        self.dssatp = String(102)
        self.das : int
        self.dynamic : int
        self.frop : int
        self.errcode : int
        self.lunio : int
        self.multi : int
        self.n_elems : int
        self.nyrs : int
        self.repno : int
        self.rotnum : int
        self.run : int
        self.trtnum : int
        self.yrdif : int
        self.yrdoy : int
        self.yrsim : int

@dataclass
class SwitchType:
    def __init__(self):
        self.fname = String(1)
        self.idetc = String(1)
        self.idetd = String(1)
        self.idetg = String(1)
        self.ideth = String(1)
        self.idetl = String(1)
        self.idetn = String(1)
        self.ideto = String(1)
        self.idetp = String(1)
        self.idetr = String(1)
        self.idets = String(1)
        self.idetw = String(1)
        self.ihari = String(1)
        self.iplti = String(1)
        self.iirri = String(1)
        self.isimi = String(1)
        self.iswche = String(1)
        self.iswdis = String(1)
        self.iswnit = String(1)
        self.iswpho = String(1)
        self.iswpot = String(1)
        self.iswsym = String(1)
        self.iswtil = String(1)
        self.iswwat = String(1)
        self.meevp = String(1)
        self.meghg = String(1)
        self.mehyd = String(1)
        self.meinf = String(1)
        self.meli = String(1)
        self.mepho = String(1)
        self.mesom = String(1)
        self.mesol = String(1)
        self.mesev = String(1)
        self.mewth = String(1)
        self.metmp = String(1)
        self.iferi = String(1)
        self.iresi = String(1)
        self.ico2 = String(1)
        self.fmopt = String(1)
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

def get_control(control_arg: List[controltype]):
    control_arg[0] = save_data.control
    

def put_control(control_arg: List[controltype]):
    save_data.control = control_arg[0]
    

def get_iswitch(iswitch_arg: List[switchtype]):
    iswitch_arg[0] = save_data.iswitch
    

def put_iswitch(iswitch_arg: List[switchtype]):
    save_data.iswitch = iswitch_arg[0]
    
