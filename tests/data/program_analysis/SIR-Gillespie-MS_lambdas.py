from numbers import Real
from random import random
import delphi.translators.for2py.math_ext as math

def SIR_Gillespie_MS__model__assign__beta__0(rho: Real, gamma: Real):
    return (rho*gamma)

def SIR_Gillespie_MS__model__assign__rateinfect__0(beta: Real, s: int, i: int, r: int):
    return (((beta*s)*i)/((s+i)+r))

def SIR_Gillespie_MS__model__assign__raterecover__0(gamma: Real, i: int):
    return (gamma*i)

def SIR_Gillespie_MS__model__assign__totalrates__0(rateinfect: Real, raterecover: Real):
    return (rateinfect+raterecover)

def SIR_Gillespie_MS__model__condition__IF_0__0(rateinfect: Real, totalrates: Real):
    return (random() < (rateinfect/totalrates))

def SIR_Gillespie_MS__model__assign__s__0(s: int):
    return (s-1)

def SIR_Gillespie_MS__model__assign__i__0(i: int):
    return (i+1)

def SIR_Gillespie_MS__model__assign__i__1(i: int):
    return (i-1)

def SIR_Gillespie_MS__model__assign__r__0(r: int):
    return (r+1)

def SIR_Gillespie_MS__model__decision__s__1(s_0: int, s_1: int, IF_0_0: bool):
    return s_1 if IF_0_0 else s_0

def SIR_Gillespie_MS__model__decision__r__1(r_0: int, r_1: int, IF_0_0: bool):
    return r_1 if IF_0_0 else r_0

def SIR_Gillespie_MS__model__decision__i__2(i_0: int, i_1: int, IF_0_0: bool):
    return i_1 if IF_0_0 else i_0

def SIR_Gillespie_MS__solver__assign__tmax__0():
    return 100

def SIR_Gillespie_MS__solver__assign__total_runs__0():
    return 1000

def SIR_Gillespie_MS__solver__assign__beta__0(rho: Real, gamma: Real):
    return (rho*gamma)

def SIR_Gillespie_MS__solver__assign__means__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_MS__solver__assign__meani__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_MS__solver__assign__meanr__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_MS__solver__assign__vars__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_MS__solver__assign__vari__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_MS__solver__assign__varr__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_MS__solver__assign__samples__0(tmax: int):
    return [0] * (0 + tmax)

def SIR_Gillespie_MS__solver__loop_0__assign__i__0():
    return 

def SIR_Gillespie_MS__solver__loop_0__condition__IF_0__0(i, tmax):
    return 0 <= i < tmax

def SIR_Gillespie_MS__solver__loop_0__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_MS__solver__loop_0__assign__means_i__0(i: int):
    means[i] = 0
    return means[i]

def SIR_Gillespie_MS__solver__loop_0__assign__meani_i__0(i: int):
    meani[i] = 0.0
    return meani[i]

def SIR_Gillespie_MS__solver__loop_0__assign__meanr_i__0(i: int):
    meanr[i] = 0.0
    return meanr[i]

def SIR_Gillespie_MS__solver__loop_0__assign__vars_i__0(i: int):
    vars[i] = 0.0
    return vars[i]

def SIR_Gillespie_MS__solver__loop_0__assign__vari_i__0(i: int):
    vari[i] = 0.0
    return vari[i]

def SIR_Gillespie_MS__solver__loop_0__assign__varr_i__0(i: int):
    varr[i] = 0.0
    return varr[i]

def SIR_Gillespie_MS__solver__loop_0__assign__samples_i__0(i: int):
    samples[i] = i
    return samples[i]

def SIR_Gillespie_MS__solver__loop_0__assign_i__1(i):
    return i + 1

def SIR_Gillespie_MS__solver__loop_1__assign__runs__0():
    return 

def SIR_Gillespie_MS__solver__loop_1__condition__IF_0__0(runs, total_runs):
    return 0 <= runs < total_runs

def SIR_Gillespie_MS__solver__loop_1__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_MS__solver__loop_1__assign__t__0():
    return 0.0

def SIR_Gillespie_MS__solver__loop_1__assign__sample_idx__0():
    return 0

def SIR_Gillespie_MS__solver__loop_1__loop_2__condition__IF_0__0(t, tmax, i):
    return ((t <= tmax) and (i > 0))

def SIR_Gillespie_MS__solver__loop_1__loop_2__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_MS__solver__loop_1__loop_2__assign__n_s__0(s: int):
    return s

def SIR_Gillespie_MS__solver__loop_1__loop_2__assign__n_i__0(i: int):
    return i

def SIR_Gillespie_MS__solver__loop_1__loop_2__assign__n_r__0(r: int):
    return r

def SIR_Gillespie_MS__solver__loop_1__loop_2__assign__dt__0(totalrates: Real):
    return -((math.log((1.0-random()))/totalrates))

def SIR_Gillespie_MS__solver__loop_1__loop_2__assign__t__0(t: Real, dt: Real):
    return (t+dt)

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__condition__IF_0__0(sample_idx, tmax, t, samples):
    return ((sample_idx < tmax) and (t > samples[sample_idx]))

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__samp__0(sample_idx: int):
    return samples[sample_idx]

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__runs1__0(runs: int):
    return (runs+1)

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__means_samp__0(samp: int, means, n_s: int, runs1: int):
    means[samp] = (means[samp]+((n_s-means[samp])/runs1))
    return means[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__vars_samp__0(samp: int, vars, runs: int, runs1: int, n_s: int, means):
    vars[samp] = (vars[samp]+(((runs/runs1)*(n_s-means[samp]))*(n_s-means[samp])))
    return vars[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__meani_samp__0(samp: int, meani, n_i: int, runs1: int):
    meani[samp] = (meani[samp]+((n_i-meani[samp])/runs1))
    return meani[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__vari_samp__0(samp: int, vari, runs: int, runs1: int, n_i: int, meani):
    vari[samp] = (vari[samp]+(((runs/runs1)*(n_i-meani[samp]))*(n_i-meani[samp])))
    return vari[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__meanr_samp__0(samp: int, meanr, n_r: int, runs1: int):
    meanr[samp] = (meanr[samp]+((n_r-meanr[samp])/runs1))
    return meanr[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__varr_samp__0(samp: int, varr, runs: int, runs1: int, n_r: int, meanr):
    varr[samp] = (varr[samp]+(((runs/runs1)*(n_r-meanr[samp]))*(n_r-meanr[samp])))
    return varr[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_2__loop_3__assign__sample_idx__0(sample_idx: int):
    return (sample_idx+1)

def SIR_Gillespie_MS__solver__loop_1__loop_4__condition__IF_0__0(sample_idx, tmax):
    return (sample_idx < tmax)

def SIR_Gillespie_MS__solver__loop_1__loop_4__decision__EXIT__0(IF_0_0):
    return IF_0_0

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__samp__0(sample_idx: int):
    return samples[sample_idx]

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__runs1__0(runs: int):
    return (runs+1)

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__means_samp__0(samp: int, means, n_s: int, runs1: int):
    means[samp] = (means[samp]+((n_s-means[samp])/runs1))
    return means[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__vars_samp__0(samp: int, vars, runs: int, runs1: int, n_s: int, means):
    vars[samp] = (vars[samp]+(((runs/runs1)*(n_s-means[samp]))*(n_s-means[samp])))
    return vars[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__meani_samp__0(samp: int, meani, n_i: int, runs1: int):
    meani[samp] = (meani[samp]+((n_i-meani[samp])/runs1))
    return meani[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__vari_samp__0(samp: int, vari, runs: int, runs1: int, n_i: int, meani):
    vari[samp] = (vari[samp]+(((runs/runs1)*(n_i-meani[samp]))*(n_i-meani[samp])))
    return vari[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__meanr_samp__0(samp: int, meanr, n_r: int, runs1: int):
    meanr[samp] = (meanr[samp]+((n_r-meanr[samp])/runs1))
    return meanr[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__varr_samp__0(samp: int, varr, runs: int, runs1: int, n_r: int, meanr):
    varr[samp] = (varr[samp]+(((runs/runs1)*(n_r-meanr[samp]))*(n_r-meanr[samp])))
    return varr[samp]

def SIR_Gillespie_MS__solver__loop_1__loop_4__assign__sample_idx__0(sample_idx: int):
    return (sample_idx+1)

def SIR_Gillespie_MS__solver__loop_1__assign_runs__1(runs):
    return runs + 1

def SIR_Gillespie_MS__main__assign__s__0():
    return 500

def SIR_Gillespie_MS__main__assign__i__0():
    return 10

def SIR_Gillespie_MS__main__assign__r__0():
    return 0

def SIR_Gillespie_MS__main__assign__tmax__0():
    return 100

def SIR_Gillespie_MS__main__assign__gamma__0():
    return (1.0/3.0)

def SIR_Gillespie_MS__main__assign__rho__0():
    return 2.0

