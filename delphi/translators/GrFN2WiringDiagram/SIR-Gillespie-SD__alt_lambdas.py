import math
import random


def SIR_Gillespie_SD__main__assign__S0__0():
    return 500


def SIR_Gillespie_SD__main__assign__I0__0():
    return 10


def SIR_Gillespie_SD__main__assign__R0__0():
    return 0


def SIR_Gillespie_SD__main__assign__Tmax__0():
    return 100


def SIR_Gillespie_SD__main__assign__MeanS__0(Tmax):
    return [0] * Tmax


def SIR_Gillespie_SD__main__assign__MeanI__0(Tmax):
    return [0] * Tmax


def SIR_Gillespie_SD__main__assign__MeanR__0(Tmax):
    return [0] * Tmax


def SIR_Gillespie_SD__main__assign__VarS__0(Tmax):
    return [0] * Tmax


def SIR_Gillespie_SD__main__assign__VarI__0(Tmax):
    return [0] * Tmax


def SIR_Gillespie_SD__main__assign__VarR__0(Tmax):
    return [0] * Tmax


def SIR_Gillespie_SD__gillespie__assign__Tmax__0():
    return 100


def SIR_Gillespie_SD__gillespie__assign__total_runs__0():
    return 1000


def SIR_Gillespie_SD__gillespie__assign__gamma__0():
    return 1.0/3.0


def SIR_Gillespie_SD__gillespie__assign__rho__0():
    return 2.0


def SIR_Gillespie_SD__gillespie__assign__beta__0(rho, gamma):
    return rho * gamma


def SIR_Gillespie_SD__gillespie__assign__samples__0(Tmax):
    return [0] * Tmax


def SIR_Gillespie_SD__update_mean_var__assign__Tmax__0():
    return 100


def SIR_Gillespie_SD__update_mean_var__assign__means_k__0(means, n, k, runs):
    means[k] = means[k] + (n - means[k])/(runs+1)
    return means[k]


def SIR_Gillespie_SD__update_mean_var__assign__vars_k__0(vars, means, n, k, runs):
    vars[k] = vars[k] + runs/(runs+1) * (n-means[k])*(n-means[k])
    return vars[k]


def SIR_Gillespie_SD__gillespie__loop_0__assign__i__0():
    return 0


def SIR_Gillespie_SD__gillespie__loop_0__condition__IF_0__0(i, Tmax):
    return 0 <= i < Tmax


def SIR_Gillespie_SD__gillespie__loop_0__decision__EXIT__0(IF_0_0):
    return IF_0_0


def SIR_Gillespie_SD__gillespie__loop_0__assign__MeanS_i__0(i, MeanS):
    MeanS[i] = 0
    return MeanS[i]


def SIR_Gillespie_SD__gillespie__loop_0__assign__MeanI_i__0(i, MeanI):
    MeanI[i] = 0.0
    return MeanI[i]


def SIR_Gillespie_SD__gillespie__loop_0__assign__MeanR_i__0(i, MeanR):
    MeanR[i] = 0.0
    return MeanR[i]


def SIR_Gillespie_SD__gillespie__loop_0__assign__VarS_i__0(i, VarS):
    VarS[i] = 0.0
    return VarS[i]


def SIR_Gillespie_SD__gillespie__loop_0__assign__VarI_i__0(i, VarI):
    VarI[i] = 0.0
    return VarI[i]


def SIR_Gillespie_SD__gillespie__loop_0__assign__VarR_i__0(i, VarR):
    VarR[i] = 0.0
    return VarR[i]


def SIR_Gillespie_SD__gillespie__loop_0__assign__samples_i__0(i, samples):
    samples[i] = i
    return samples[i]


def SIR_Gillespie_SD__gillespie__loop_0__assign__i__1(i):
    return i + 1


def SIR_Gillespie_SD__gillespie__loop_1__assign__runs__0():
    return 0


def SIR_Gillespie_SD__gillespie__loop_1__condition__IF_0__0(runs, total_runs):
    return 0 <= runs < total_runs-1


def SIR_Gillespie_SD__gillespie__loop_1__decision__EXIT__0(IF_0_0):
    return IF_0_0


def SIR_Gillespie_SD__gillespie__loop_1__assign__t__0():
    return 0.0


def SIR_Gillespie_SD__gillespie__loop_1__assign__n_S__0(S0):
    return S0


def SIR_Gillespie_SD__gillespie__loop_1__assign__n_I__0(I0):
    return I0


def SIR_Gillespie_SD__gillespie__loop_1__assign__n_R__0(R0):
    return R0


def SIR_Gillespie_SD__gillespie__loop_1__assign__sample_idx__0():
    return 0


def SIR_Gillespie_SD__gillespie__loop_1__assign__runs__1(runs):
    return runs + 1


def SIR_Gillespie_SD__gillespie__loop_2__condition__IF_0__0(t, Tmax, n_I):
    return t <= Tmax and n_I > 0


def SIR_Gillespie_SD__gillespie__loop_2__decision__EXIT__0(IF_0_0):
    return IF_0_0


def SIR_Gillespie_SD__gillespie__loop_2__assign__rateInfect__0(beta, n_S, n_I, n_R):
    return beta * n_S * n_I / (n_S + n_I + n_R)


def SIR_Gillespie_SD__gillespie__loop_2__assign__rateRecover__0(gamma, n_I):
    return gamma * n_I


def SIR_Gillespie_SD__gillespie__loop_2__assign__totalRates__0(rateInfect, rateRecover):
    return rateInfect + rateRecover


def SIR_Gillespie_SD__gillespie__loop_2__assign__dt__0(totalRates):
    return -math.log(1.0-random())/totalRates


def SIR_Gillespie_SD__gillespie__loop_2__assign__t__0(t, dt):
    return t + dt


def SIR_Gillespie_SD__gillespie__loop_2__condition__IF_1__0(rateInfect, totalRates):
    return random() < (rateInfect/totalRates)


def SIR_Gillespie_SD__gillespie__loop_2__assign__n_S__0(n_S):
    return n_S - 1


def SIR_Gillespie_SD__gillespie__loop_2__assign__n_I__0(n_I):
    return n_I + 1


def SIR_Gillespie_SD__gillespie__loop_2__assign__n_I__1(n_I):
    return n_I - 1


def SIR_Gillespie_SD__gillespie__loop_2__assign__n_R__0(n_R):
    return n_R + 1


def SIR_Gillespie_SD__gillespie__loop_2__decision__n_I__2(n_I_0, n_I_1, IF_1):
    return n_I_1 if IF_1 else n_I_0


def SIR_Gillespie_SD__gillespie__loop_2__decision__n_S__1(n_S_0, n_S_1, IF_1):
    return n_S_1 if IF_1 else n_S_0


def SIR_Gillespie_SD__gillespie__loop_2__decision__n_R__1(n_R_0, n_R_1, IF_1):
    return n_R_1 if IF_1 else n_R_0


def SIR_Gillespie_SD__gillespie__loop_3__condition__IF_0__0(sample_idx, Tmax, t, samples):
    return sample_idx < Tmax and t > samples(sample_idx)


def SIR_Gillespie_SD__gillespie__loop_3__decision__EXIT__0(IF_0_0):
    return IF_0_0


def SIR_Gillespie_SD__gillespie__loop_3__assign__sample__0(samples, sample_idx):
    return samples[sample_idx]


def SIR_Gillespie_SD__gillespie__loop_3__assign__sample_idx__0(sample_idx):
    return sample_idx + 1


def SIR_Gillespie_SD__gillespie__loop_4__condition__IF_0__0(sample_idx, Tmax):
    return sample_idx < Tmax


def SIR_Gillespie_SD__gillespie__loop_4__decision__EXIT__0(IF_0_0):
    return IF_0_0


def SIR_Gillespie_SD__gillespie__loop_4__assign__sample__0(samples, sample_idx):
    return samples[sample_idx]


def SIR_Gillespie_SD__gillespie__loop_4__assign__sample_idx__0(sample_idx):
    return sample_idx + 1
