import math
import random
import matplotlib.pyplot as plt
""" SIR Model implemented as a Monte Carlo Simulation via the Gillespie Method
State Variables: S, I, R
    S - Susceptible population
    I - Infected population
    R - Recovered population
Events: Infection, Recovery
"""

# int: Initial value of S
S0 = 500
# int: Initial value of I
I0 = 10
# int: Initial value of R
R0 = 0

# float: Maximum time for the simulation
T = 100.0

# float: Initial time for the simulation
t = 0.0

# int: Total number of trajectories to generate for the analysis
totalRuns = 1000

# float: Rate of recovery from an infection
gamma = 1.0 / 3.0

# float: Basic reproduction Number
rho = 2.0

# float: Rate of infection
beta = rho * gamma

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

# Main loop over totalRuns iterations
for runs in range(0, totalRuns):
    # int array: Time points to collect measurements of mean and variance
    samples = [x for x in range(0, int(math.ceil(T)))]

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
        rateInfect = beta * float(n_S) * float(n_I) / float(n_S + n_I + n_R)
        # float: Current state dependent rate of recovery
        rateRecover = gamma * float(n_I)
        # Sum of total rates; taking advantage of Markovian identities to improve
        # performance.
        totalRates = rateInfect + rateRecover

        # float: next inter-event time
        dt = -math.log(1.0 - random.uniform(0.0, 1.0)) / totalRates

        # Advance the system clock
        t = t + dt

        # Calculate all measures up to the current time t
        while sample_idx < len(samples) and t > float(samples[sample_idx]):
            sample = samples[sample_idx]
            # Welford's one pass algorithm for mean and variance
            MeanS[sample] += (float(n_S) - MeanS[sample]) / float(runs+1)
            MeanI[sample] += (float(n_I) - MeanI[sample]) / float(runs+1)
            MeanR[sample] += (float(n_R) - MeanR[sample]) / float(runs+1)
            VarS[sample] += float(runs)/float(runs + 1) * (float(n_S) - MeanS[sample])*(float(n_S) - MeanS[sample])
            VarI[sample] += float(runs)/float(runs + 1) * (float(n_I) - MeanI[sample])*(float(n_I) - MeanI[sample])
            VarR[sample] += float(runs)/float(runs + 1) * (float(n_R) - MeanR[sample])*(float(n_R) - MeanR[sample])
            sample_idx += 1

        # Determine which event fired.  With probability rateInfect/totalRates
        # the next event is infection.
        if random.uniform(0.0,1.0) < (rateInfect / totalRates):
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
        MeanS[sample] += (float(n_S) - MeanS[sample]) / float(runs+1)
        MeanI[sample] += (float(n_I) - MeanI[sample]) / float(runs+1)
        MeanR[sample] += (float(n_R) - MeanR[sample]) / float(runs+1)
        VarS[sample] += float(runs)/float(runs + 1) * (float(n_S) - MeanS[sample])*(float(n_S) - MeanS[sample])
        VarI[sample] += float(runs)/float(runs + 1) * (float(n_I) - MeanI[sample])*(float(n_I) - MeanI[sample])
        VarR[sample] += float(runs)/float(runs + 1) * (float(n_R) - MeanR[sample])*(float(n_R) - MeanR[sample])
        sample_idx += 1

# float array: Time points for all measures for plotting
t = [float(x) for x in range(0, int(math.ceil(T)))]

# float: Unbiased variance computation for S
VarS = [v/float(totalRuns) for v in VarS]
# float: Unbiased variance computation for I
VarI = [v/float(totalRuns) for v in VarI]
# float: Unbiased variance computation for R
VarR = [v/float(totalRuns) for v in VarR]

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(t, MeanS, label="S")
# float: negative and positive confidence intervals for E[S]
nCI = [x-y for x,y in zip(MeanS, [1.96 * math.sqrt(x)/math.sqrt(totalRuns) for x in VarS])]
pCI = [x+y for x,y in zip(MeanS, [1.96 * math.sqrt(x)/math.sqrt(totalRuns) for x in VarS])]
ax.fill_between(t, nCI, pCI, alpha=.1)
ax.plot(t, MeanI, label="I")
# float: negative and positive confidence intervals for E[I]
nCI = [x-y for x,y in zip(MeanI, [1.96 * math.sqrt(x)/math.sqrt(totalRuns) for x in VarI])]
pCI = [x+y for x,y in zip(MeanI, [1.96 * math.sqrt(x)/math.sqrt(totalRuns) for x in VarI])]
ax.fill_between(t, nCI, pCI, alpha=.1)
ax.plot(t, MeanR, label="R")
# float: negative and positive confidence intervals for E[R]
nCI = [x-y for x,y in zip(MeanR, [1.96 * math.sqrt(x)/math.sqrt(totalRuns) for x in VarR])]
pCI = [x+y for x,y in zip(MeanR, [1.96 * math.sqrt(x)/math.sqrt(totalRuns) for x in VarR])]
ax.fill_between(t, nCI, pCI, alpha=.1)
ax.legend()
fig.savefig("Python-Gillespie.pdf", bbox_inches='tight')
