from numbers import Real
from random import random
import delphi.translators.for2py.math_ext as math

def SIR_Gillespie_SD_inline__gillespie__assign__tmax__0():
    return 100

def SIR_Gillespie_SD_inline__gillespie__assign__total_runs__0():
    return 1000

def SIR_Gillespie_SD_inline__gillespie__assign__gamma__0():
    return (1.0/3.0)

def SIR_Gillespie_SD_inline__gillespie__assign__rho__0():
    return 2.0

def SIR_Gillespie_SD_inline__gillespie__assign__beta__0(rho: Real, gamma: Real):
    return (rho*gamma)

def SIR_Gillespie_SD_inline__gillespie__assign__means__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_SD_inline__gillespie__assign__meani__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_SD_inline__gillespie__assign__meanr__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_SD_inline__gillespie__assign__vars__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_SD_inline__gillespie__assign__vari__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_SD_inline__gillespie__assign__varr__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_SD_inline__gillespie__assign__samples__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__i__0():
    return 0

def SIR_Gillespie_SD_inline__gillespie__loop_0__condition__IF_0__0(i, tmax):
    return 0 <= i < tmax

def SIR_Gillespie_SD_inline__gillespie__loop_0__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__means_i__0(i: int):
    means[i] = 0
    return means[i]

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__meani_i__0(i: int):
    meani[i] = 0.0
    return meani[i]

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__meanr_i__0(i: int):
    meanr[i] = 0.0
    return meanr[i]

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__vars_i__0(i: int):
    vars[i] = 0.0
    return vars[i]

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__vari_i__0(i: int):
    vari[i] = 0.0
    return vari[i]

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__varr_i__0(i: int):
    varr[i] = 0.0
    return varr[i]

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign__samples_i__0(i: int):
    samples[i] = i
    return samples[i]

def SIR_Gillespie_SD_inline__gillespie__loop_0__assign_i__1(i):
    return i + 1

def SIR_Gillespie_SD_inline__gillespie__assign__n_samples__0():
    return 0

def SIR_Gillespie_SD_inline__gillespie__loop_1__assign__runs__0():
    return 0

def SIR_Gillespie_SD_inline__gillespie__loop_1__condition__IF_0__0(runs, total_runs):
    return 0 <= runs < total_runs

def SIR_Gillespie_SD_inline__gillespie__loop_1__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_SD_inline__gillespie__loop_1__assign__t__0():
    return 0.0

def SIR_Gillespie_SD_inline__gillespie__loop_1__assign__n_s__0(s0: int):
    return s0

def SIR_Gillespie_SD_inline__gillespie__loop_1__assign__n_i__0(i0: int):
    return i0

def SIR_Gillespie_SD_inline__gillespie__loop_1__assign__n_r__0(r0: int):
    return r0

def SIR_Gillespie_SD_inline__gillespie__loop_1__assign__sample_idx__0():
    return 0

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__condition__IF_0__0(t, tmax, n_i):
    return ((t <= tmax) and (n_i > 0))

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__rateinfect__0(beta: Real, n_s: int, n_i: int, n_r: int):
    return (((beta*n_s)*n_i)/((n_s+n_i)+n_r))

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__raterecover__0(gamma: Real, n_i: int):
    return (gamma*n_i)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__totalrates__0(rateinfect: Real, raterecover: Real):
    return (rateinfect+raterecover)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__dt__0(totalrates: Real):
    return -((math.log((1.0-random()))/totalrates))

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__t__0(t: Real, dt: Real):
    return (t+dt)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__condition__IF_0__0(sample_idx, tmax, t, samples):
    return ((sample_idx < tmax) and (t > samples[sample_idx]))

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__samp__0(sample_idx: int):
    return samples[sample_idx]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__runs1__0(runs: int):
    return (runs+1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__means_samp__0(samp: int, means, n_s: int, runs1: int):
    means[samp] = (means[samp]+((n_s-means[samp])/runs1))
    return means[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__vars_samp__0(samp: int, vars, runs: int, runs1: int, n_s: int, means):
    vars[samp] = (vars[samp]+(((runs/runs1)*(n_s-means[samp]))*(n_s-means[samp])))
    return vars[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__meani_samp__0(samp: int, meani, n_i: int, runs1: int):
    meani[samp] = (meani[samp]+((n_i-meani[samp])/runs1))
    return meani[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__vari_samp__0(samp: int, vari, runs: int, runs1: int, n_i: int, meani):
    vari[samp] = (vari[samp]+(((runs/runs1)*(n_i-meani[samp]))*(n_i-meani[samp])))
    return vari[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__meanr_samp__0(samp: int, meanr, n_r: int, runs1: int):
    meanr[samp] = (meanr[samp]+((n_r-meanr[samp])/runs1))
    return meanr[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__varr_samp__0(samp: int, varr, runs: int, runs1: int, n_r: int, meanr):
    varr[samp] = (varr[samp]+(((runs/runs1)*(n_r-meanr[samp]))*(n_r-meanr[samp])))
    return varr[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__loop_3__assign__sample_idx__0(sample_idx: int):
    return (sample_idx+1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__condition__IF_0__0(rateinfect: Real, totalrates: Real):
    return (random() < (rateinfect/totalrates))

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__n_s__0(n_s: int):
    return (n_s-1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__n_i__0(n_i: int):
    return (n_i+1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__n_i__1(n_i: int):
    return (n_i-1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__assign__n_r__0(n_r: int):
    return (n_r+1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__decision__n_r__1(n_r_0: int, n_r_1: int, IF_0_0: bool):
    return n_r_1 if IF_0_0 else n_r_0

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__decision__n_i__2(n_i_0: int, n_i_1: int, IF_0_0: bool):
    return n_i_1 if IF_0_0 else n_i_0

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_2__decision__n_s__1(n_s_0: int, n_s_1: int, IF_0_0: bool):
    return n_s_1 if IF_0_0 else n_s_0

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__condition__IF_0__0(sample_idx, tmax):
    return (sample_idx < tmax)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__samp__0(sample_idx: int):
    return samples[sample_idx]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__runs1__0(runs: int):
    return (runs+1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__means_samp__0(samp: int, means, n_s: int, runs1: int):
    means[samp] = (means[samp]+((n_s-means[samp])/runs1))
    return means[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__vars_samp__0(samp: int, vars, runs: int, runs1: int, n_s: int, means):
    vars[samp] = (vars[samp]+(((runs/runs1)*(n_s-means[samp]))*(n_s-means[samp])))
    return vars[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__meani_samp__0(samp: int, meani, n_i: int, runs1: int):
    meani[samp] = (meani[samp]+((n_i-meani[samp])/runs1))
    return meani[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__vari_samp__0(samp: int, vari, runs: int, runs1: int, n_i: int, meani):
    vari[samp] = (vari[samp]+(((runs/runs1)*(n_i-meani[samp]))*(n_i-meani[samp])))
    return vari[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__meanr_samp__0(samp: int, meanr, n_r: int, runs1: int):
    meanr[samp] = (meanr[samp]+((n_r-meanr[samp])/runs1))
    return meanr[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__varr_samp__0(samp: int, varr, runs: int, runs1: int, n_r: int, meanr):
    varr[samp] = (varr[samp]+(((runs/runs1)*(n_r-meanr[samp]))*(n_r-meanr[samp])))
    return varr[samp]

def SIR_Gillespie_SD_inline__gillespie__loop_1__loop_4__assign__sample_idx__0(sample_idx: int):
    return (sample_idx+1)

def SIR_Gillespie_SD_inline__gillespie__loop_1__assign_runs__1(runs):
    return runs + 1

def SIR_Gillespie_SD_inline__main__assign__s0__0():
    return 500

def SIR_Gillespie_SD_inline__main__assign__i0__0():
    return 10

def SIR_Gillespie_SD_inline__main__assign__r0__0():
    return 0

def SIR_Gillespie_SD_inline__main__assign__tmax__0():
    return 100

