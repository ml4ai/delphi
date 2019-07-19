""" SIR Model implemented as a Monte Carlo Simulation via the Gillespie Method
State Variables: S, I, R
    S - Susceptible population
    I - Infected population
    R - Recovered population
Events: Infection, Recovery
"""

import math
import random

def print_output(L):
    n = 5   # no. of values per line
    for i in range(len(L)//n):
        print("{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".\
              format(L[n*i+0], L[n*i+1], L[n*i+2], L[n*i+3], L[n*i+4]))
    print("----------")


# Welford's one pass algorithm for mean and variance
def update_mean_var(means, variances, k, n, runs):
    upd_mean = (n - means[k])/(runs+1)
    means[k] += upd_mean

    upd_var = runs/(runs+1) * (n-means[k])*(n-means[k])
    variances[k] += upd_var


def main():
    S0 = 500    # int: Initial value of S
    I0 = 10     # int: Initial value of I
    R0 = 0      # int: Initial value of R
    T = 100     # float: Maximum time for the simulation
    t = 0.0     # float: Initial time for the simulation
    
    totalRuns = 1000    # int: Total number of trajectories to generate for the analysis
    gamma = 1.0 / 3.0   # float: Rate of recovery from an infection
    rho = 2.0    # float: Basic reproduction Number
    beta = rho * gamma  # float: Rate of infection
    
    # float array: Measures of Mean for S, I, and R
    MeanS = [0.0] * int(math.ceil(T))
    MeanI = [0.0] * int(math.ceil(T))
    MeanR = [0.0] * int(math.ceil(T))
    
    # float array: Measures of Variance for S, I, and R
    VarS = [0.0] * int(math.ceil(T))
    VarI = [0.0] * int(math.ceil(T))
    VarR = [0.0] * int(math.ceil(T))
    
    # int: Total samples collected so far, used for Welford's one pass algorithm
    nSamples = 0
    
    random.seed(0)    # pick an arbitrary value; I chose 0

    # int array: Time points to collect measurements of mean and variance
    samples = [x for x in range(0, int(math.ceil(T)))]

    # Main loop over totalRuns iterations
    for runs in range(0, totalRuns):
        # Restart the event clock
        t = 0.0
    
        # Set the initial conditions of S, I, and R
        # int: n_S - current number of Susceptible
        # int: n_I - current number of Infected
        # int: n_R - current number of Recovered
        (n_S, n_I, n_R) = (S0, I0, R0)
    
        # Main Gillespie Loop
        sample_idx = 0
        while (t < T) and (n_I > 0.0):
            # float: Current state dependent rate of infection
            rateInfect = beta * n_S * n_I / (n_S + n_I + n_R)
            # float: Current state dependent rate of recovery
            rateRecover = gamma * n_I
            # Sum of total rates; taking advantage of Markovian identities to improve performance.
            totalRates = rateInfect + rateRecover

            # float: next inter-event time
            randval = random.uniform(0.0, 1.0)
            dt = -math.log(1.0 - randval) / totalRates

            # Advance the system clock
            t = t + dt
    
            # Calculate all measures up to the current time t using
            # Welford's one pass algorithm for mean and variance
            while sample_idx < len(samples) and t > samples[sample_idx]:
                sample = samples[sample_idx]
                update_mean_var(MeanS, VarS, sample, n_S, runs)
                update_mean_var(MeanI, VarI, sample, n_I, runs)
                update_mean_var(MeanR, VarR, sample, n_R, runs)
                sample_idx += 1

            # Determine which event fired.  With probability rateInfect/totalRates
            # the next event is infection.
            randval = random.uniform(0.0, 1.0)
            if randval < (rateInfect / totalRates):
                # Delta for infection
                n_S = n_S - 1
                n_I = n_I + 1
    
            # Determine the event fired.  With probability rateRecover/totalRates
            # the next event is recovery.
            else:
                # Delta for recovery
                n_I = n_I - 1
                n_R = n_R + 1

        # After all events have been processed, clean up by evaluating all remaining measures.
        while sample_idx < len(samples):
            sample = samples[sample_idx]
            update_mean_var(MeanS, VarS, sample, n_S, runs)
            update_mean_var(MeanI, VarI, sample, n_I, runs)
            update_mean_var(MeanR, VarR, sample, n_R, runs)
            sample_idx += 1

    print_output(MeanS)
    print_output(MeanI)
    print_output(MeanR)

#    VarS = [v/totalRuns for v in VarS]    # Unbiased variance computation for S
#    VarI = [v/totalRuns for v in VarI]    # Unbiased variance computation for I
#    VarR = [v/totalRuns for v in VarR]    # Unbiased variance computation for R
#    
#    print_output(VarS)
#    print_output(VarI)
#    print_output(VarR)

main()

