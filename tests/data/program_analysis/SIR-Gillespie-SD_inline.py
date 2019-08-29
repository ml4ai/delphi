import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real


@static_vars([{'name': 'file_2', 'call': None, 'type': 'file_handle'}])
def pyrand(do_init: List[bool]):
    retval: List[float] = [None]
    if (do_init[0] == True):
        pyrand.file_2 = open("PY_RANDOM_GILLESPIE", "r")
        retval[0] = 0.0
    else:
        format_10: List[str] = [None]
        format_10 = ['F20.18']
        format_10_obj = Format(format_10)
        
        (retval[0],) = format_10_obj.read_line(pyrand.file_2.readline())
        
        
    return retval[0]

def gillespie(s0: List[int], i0: List[int], r0: List[int]):
    tmax: List[int] = [100]
    total_runs: List[int] = [1000]
    gamma: List[float] = [(1.0 / 3.0)]
    rho: List[float] = [2.0]
    beta: List[float] = [(rho[0] * gamma[0])]
    means = Array(float, [(0, tmax[0])])
    meani = Array(float, [(0, tmax[0])])
    meanr = Array(float, [(0, tmax[0])])
    vars = Array(float, [(0, tmax[0])])
    vari = Array(float, [(0, tmax[0])])
    varr = Array(float, [(0, tmax[0])])
    samples = Array(int, [(0, tmax[0])])
    i: List[int] = [None]
    n_samples: List[int] = [None]
    runs: List[int] = [None]
    n_s: List[int] = [None]
    n_i: List[int] = [None]
    n_r: List[int] = [None]
    sample_idx: List[int] = [None]
    samp: List[int] = [None]
    runs1: List[int] = [None]
    t: List[float] = [None]
    randval: List[float] = [None]
    rateinfect: List[float] = [None]
    raterecover: List[float] = [None]
    totalrates: List[float] = [None]
    dt: List[float] = [None]
    for i[0] in range(0, tmax[0]+1):
        means.set_((i[0]), 0)
        meani.set_((i[0]), 0.0)
        meanr.set_((i[0]), 0.0)
        vars.set_((i[0]), 0.0)
        vari.set_((i[0]), 0.0)
        varr.set_((i[0]), 0.0)
        samples.set_((i[0]), i[0])
    n_samples[0] = 0
    randval[0] = pyrand([True])
    for runs[0] in range(0, (total_runs[0] - 1)+1):
        t[0] = 0.0
        n_s[0] = s0[0]
        n_i[0] = i0[0]
        n_r[0] = r0[0]
        sample_idx[0] = 0
        while ((t[0] <= tmax[0]) and (n_i[0] > 0)):
            rateinfect[0] = (((beta[0] * n_s[0]) * n_i[0]) / ((n_s[0] + n_i[0]) + n_r[0]))
            raterecover[0] = (gamma[0] * n_i[0])
            totalrates[0] = (rateinfect[0] + raterecover[0])
            randval[0] = pyrand([False])
            dt[0] = -((math.log((1.0 - randval[0])) / totalrates[0]))
            t[0] = (t[0] + dt[0])
            while ((sample_idx[0] < tmax[0]) and (t[0] > samples.get_((sample_idx[0])))):
                samp[0] = samples.get_((sample_idx[0]))
                runs1[0] = (runs[0] + 1)
                means.set_((samp[0]), (means.get_((samp[0])) + ((n_s[0] - means.get_((samp[0]))) / runs1[0])))
                vars.set_((samp[0]), (vars.get_((samp[0])) + (((runs[0] / runs1[0]) * (n_s[0] - means.get_((samp[0])))) * (n_s[0] - means.get_((samp[0]))))))
                meani.set_((samp[0]), (meani.get_((samp[0])) + ((n_i[0] - meani.get_((samp[0]))) / runs1[0])))
                vari.set_((samp[0]), (vari.get_((samp[0])) + (((runs[0] / runs1[0]) * (n_i[0] - meani.get_((samp[0])))) * (n_i[0] - meani.get_((samp[0]))))))
                meanr.set_((samp[0]), (meanr.get_((samp[0])) + ((n_r[0] - meanr.get_((samp[0]))) / runs1[0])))
                varr.set_((samp[0]), (varr.get_((samp[0])) + (((runs[0] / runs1[0]) * (n_r[0] - meanr.get_((samp[0])))) * (n_r[0] - meanr.get_((samp[0]))))))
                sample_idx[0] = (sample_idx[0] + 1)
            randval[0] = pyrand([False])
            if (randval[0] < (rateinfect[0] / totalrates[0])):
                n_s[0] = (n_s[0] - 1)
                n_i[0] = (n_i[0] + 1)
            else:
                n_i[0] = (n_i[0] - 1)
                n_r[0] = (n_r[0] + 1)
        while (sample_idx[0] < tmax[0]):
            samp[0] = samples.get_((sample_idx[0]))
            runs1[0] = (runs[0] + 1)
            means.set_((samp[0]), (means.get_((samp[0])) + ((n_s[0] - means.get_((samp[0]))) / runs1[0])))
            vars.set_((samp[0]), (vars.get_((samp[0])) + (((runs[0] / runs1[0]) * (n_s[0] - means.get_((samp[0])))) * (n_s[0] - means.get_((samp[0]))))))
            meani.set_((samp[0]), (meani.get_((samp[0])) + ((n_i[0] - meani.get_((samp[0]))) / runs1[0])))
            vari.set_((samp[0]), (vari.get_((samp[0])) + (((runs[0] / runs1[0]) * (n_i[0] - meani.get_((samp[0])))) * (n_i[0] - meani.get_((samp[0]))))))
            meanr.set_((samp[0]), (meanr.get_((samp[0])) + ((n_r[0] - meanr.get_((samp[0]))) / runs1[0])))
            varr.set_((samp[0]), (varr.get_((samp[0])) + (((runs[0] / runs1[0]) * (n_r[0] - meanr.get_((samp[0])))) * (n_r[0] - meanr.get_((samp[0]))))))
            sample_idx[0] = (sample_idx[0] + 1)

def main():
    s0: List[int] = [500]
    i0: List[int] = [10]
    r0: List[int] = [0]
    tmax: List[int] = [100]
    gillespie(s0, i0, r0)

main()
