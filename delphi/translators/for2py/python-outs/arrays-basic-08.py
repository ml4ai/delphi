import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    format_10: List[str] = []
    format_10 = ['/', "' GAUSS ELIMINATION'"]
    format_10_obj = Format(format_10)
    
    format_11: List[str] = []
    format_11 = ['5(X,F8.2)']
    format_11_obj = Format(format_11)
    
    format_12: List[str] = []
    format_12 = ['/', "' AUGMENTED MATRIX'", '/']
    format_12_obj = Format(format_12)
    
    format_61: List[str] = []
    format_61 = ['5(1X,f8.4)']
    format_61_obj = Format(format_61)
    
    format_13: List[str] = []
    format_13 = ["''"]
    format_13_obj = Format(format_13)
    
    format_14: List[str] = []
    format_14 = ["' SOLUTION'"]
    format_14_obj = Format(format_14)
    
    format_15: List[str] = []
    format_15 = ["' ...........................................'"]
    format_15_obj = Format(format_15)
    
    format_16: List[str] = []
    format_16 = ["'         I       X(I)'"]
    format_16_obj = Format(format_16)
    
    format_72: List[str] = []
    format_72 = ['5X', 'I5', 'F12.6']
    format_72_obj = Format(format_72)
    
    a = Array(float, [(1, 20), (1, 21)])
    i: List[int] = [0]
    j: List[int] = [0]
    n: List[int] = [0]
    
    write_list_stream = []
    write_line = format_10_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    n[0] = 4
    
    file_10 = open("INFILE-GAUSSIAN", "r")
    for i[0] in range(1, n[0]+1):
        
        
        
        
        tempVar = [0] * 5
        (tempVar[0], tempVar[1], tempVar[2], tempVar[3], tempVar[4]) = format_11_obj.read_line(file_10.readline())
        a.set_((i[0], 1), tempVar[0])
        a.set_((i[0], 2), tempVar[1])
        a.set_((i[0], 3), tempVar[2])
        a.set_((i[0], 4), tempVar[3])
        a.set_((i[0], 5), tempVar[4])
        
    file_10.close()
    
    write_list_stream = []
    write_line = format_12_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    
    for i[0] in range(1, n[0]+1):
        write_list_stream = [a.get_((i[0], 1)), a.get_((i[0], 2)), a.get_((i[0], 3)), a.get_((i[0], 4)), a.get_((i[0], 5))]
        write_line = format_61_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    
    
    
    
    write_list_stream = []
    write_line = format_13_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    gauss(n, a)
    write_list_stream = []
    write_line = format_13_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_14_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_15_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_16_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_15_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    
    for i[0] in range(1, n[0]+1):
        write_list_stream = [i[0], a.get_((i[0], n[0] + 1 ))]
        write_line = format_72_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_15_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    write_list_stream = []
    write_line = format_13_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    return

def gauss(n: List[int], a):
    format_11: List[str] = []
    format_11 = ["'      MACHINE EPSILON='", 'E16.8']
    format_11_obj = Format(format_11)
    
    format_12: List[str] = []
    format_12 = ['/', "'  DETERMINANT= '", 'F16.5', '/']
    format_12_obj = Format(format_12)
    
    
    pv: List[int] = [0]
    i: List[int] = [0]
    j: List[int] = [0]
    k: List[int] = [0]
    r: List[int] = [0]
    jc: List[int] = [0]
    jr: List[int] = [0]
    kc: List[int] = [0]
    nv: List[int] = [0]
    eps: List[float] = [0.0]
    eps2: List[float] = [0.0]
    det: List[float] = [0.0]
    tm: List[float] = [0.0]
    temp: List[float] = [0.0]
    va: List[float] = [0.0]
    eps[0] = 1.0
    while ((1.0 + eps[0]) > 1.0):
        eps[0] = (eps[0] / 2.0)
    eps[0] = (eps[0] * 2)
    
    write_list_stream = [eps[0]]
    write_line = format_11_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    eps2[0] = (eps[0] * 2)
    det[0] = 1.0
    for i[0] in range(1, (n[0] - 1)+1):
        pv[0] = i[0]
        for j[0] in range((i[0] + 1), n[0]+1):
            if (abs(a.get_(((pv[0], i[0])))) < abs(a.get_(((j[0], i[0]))))):
                pv[0] = j[0]
        if (pv[0] != i[0]):
            for jc[0] in range(1, (n[0] + 1)+1):
                tm[0] = a.get_((i[0], jc[0]))
                a.set_((i[0], jc[0]), a.get_((pv[0], jc[0])))
                a.set_((pv[0], jc[0]), tm[0])
            det[0] = -((1 * det[0]))
        if (a.get_(((i[0], i[0]))) == 0.0):
            print('MATRIX IS SINGULAR')
            return
        for jr[0] in range((i[0] + 1), n[0]+1):
            if (a.get_(((jr[0], i[0]))) != 0.0):
                r[0] = (a.get_(((jr[0], i[0]))) / a.get_(((i[0], i[0]))))
                for kc[0] in range((i[0] + 1), (n[0] + 1)+1):
                    temp[0] = a.get_((jr[0], kc[0]))
                    a.set_((jr[0], kc[0]), (a.get_(((jr[0], kc[0]))) - (r[0] * a.get_(((i[0], kc[0]))))))
                    if (abs(a.get_(((jr[0], kc[0])))) < (eps2[0] * temp[0])):
                        a.set_((jr[0], kc[0]), 0.0)
    for i[0] in range(1, n[0]+1):
        det[0] = (det[0] * a.get_(((i[0], i[0]))))
    
    write_list_stream = [det[0]]
    write_line = format_12_obj.write_line(write_list_stream)
    sys.stdout.write(write_line)
    if (a.get_(((n[0], n[0]))) == 0.0):
        print('MATRIX IS SINGULAR')
        return
    a.set_((n[0], n[0] + 1), (a.get_(((n[0], n[0]+1))) / a.get_(((n[0], n[0])))))
    for nv[0] in range((n[0] - 1), 1+1, -(1)):
        va[0] = a.get_((nv[0], n[0] + 1))
        for k[0] in range((nv[0] + 1), n[0]+1):
            va[0] = (va[0] - (a.get_(((nv[0], k[0]))) * a.get_(((k[0], n[0]+1)))))
        a.set_((nv[0], n[0] + 1), (va[0] / a.get_(((nv[0], nv[0])))))
    
    

main()
