import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass
from m_mymod9 import *


def pgm():
    format_10: List[str] = []
    format_10 = ['5(I6,2X)']
    format_10_obj = Format(format_10)
    
    format_15: List[str] = []
    format_15 = ['2(F8.4,X)', '5(I3,X)']
    format_15_obj = Format(format_15)
    
    
    
    
    write_list_stream = [nl[0], ts[0], nappl[0], ncohorts[0], nelem[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = [numofdays[0], nstalks[0], evalnum[0], maxfiles[0], maxpest[0]]
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = [pi[0], rad[0], runinit[0], init[0], seasinit[0], rate[0], emerg[0]]
    write_line = format_15_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

pgm()
