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
from random import random


def gillespie(s: List[int], i: List[int], r: List[int], gamma: List[float], rho: List[float]):
    tmax: List[int] = [100]
    total_runs: List[int] = [1000]
    beta: List[float] = [(rho[0] * gamma[0])]
    means = Array(float, [(0, tmax[0])])
    meani = Array(float, [(0, tmax[0])])
    meanr = Array(float, [(0, tmax[0])])
    vars = Array(float, [(0, tmax[0])])
    vari = Array(float, [(0, tmax[0])])
    varr = Array(float, [(0, tmax[0])])
    samples = Array(int, [(0, tmax[0])])
    j: List[int] = [None]
    runs: List[int] = [None]
    n_s: List[int] = [None]
    n_i: List[int] = [None]
    n_r: List[int] = [None]
    sample_idx: List[int] = [None]
    samp: List[int] = [None]
    runs1: List[int] = [None]
    totalrates: List[float] = [None]
    dt: List[float] = [None]
    t: List[float] = [None]
    randval: List[float] = [None]
    rateinfect: List[float] = [None]
    raterecover: List[float] = [None]
    for j[0] in range(0, tmax[0]+1):
        means.set_((j[0]), 0)
        meani.set_((j[0]), 0.0)
        meanr.set_((j[0]), 0.0)
        vars.set_((j[0]), 0.0)
        vari.set_((j[0]), 0.0)
        varr.set_((j[0]), 0.0)
        samples.set_((j[0]), j[0])
    for runs[0] in range(0, (total_runs[0] - 1)+1):
        t[0] = 0.0
        sample_idx[0] = 0
        while ((t[0] <= tmax[0]) and (i[0] > 0)):
            n_s[0] = s[0]
            n_i[0] = i[0]
            n_r[0] = r[0]
            rateinfect[0] = (((beta[0] * s[0]) * i[0]) / ((s[0] + i[0]) + r[0]))
            raterecover[0] = (gamma[0] * i[0])
            totalrates[0] = (rateinfect[0] + raterecover[0])
            if (random() < (rateinfect[0] / totalrates[0])):
                s[0] = (s[0] - 1)
                i[0] = (i[0] + 1)
            else:
                i[0] = (i[0] - 1)
                r[0] = (r[0] + 1)
            dt[0] = -((math.log((1.0 - random())) / totalrates[0]))
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
    s: List[int] = [500]
    i: List[int] = [10]
    r: List[int] = [0]
    tmax: List[int] = [100]
    gamma: List[float] = [(1.0 / 3.0)]
    rho: List[float] = [2.0]
    gillespie(s, i, r, gamma, rho)

main()
