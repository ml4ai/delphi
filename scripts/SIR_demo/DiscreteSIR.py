from random import random, seed
import math

import numpy as np


# A function for computing the updated amounts of infected and recovered
def randbn(n, p):
    q = 1.0 - p
    s = p/q
    a = (n+1)*s
    r = math.exp(n*math.log(q))
    x = 0
    u = random()
    while True:
        if u < r:
            return x
        u -= r
        x += 1
        r *= (a/x)-s


# Actual model code here, this should be interchangeable with SEIR
def sir(u, prams):
    (S, I, R, Y) = u
    (β, γ, ι, N, δt) = prams
    λ = β * (I + ι) / N
    ifrac = 1.0 - math.exp(-λ * δt)
    rfrac = 1.0 - math.exp(-γ * δt)
    infection = randbn(S, ifrac)
    recovery = randbn(I, rfrac)
    return (S - infection, I + infection - recovery, R + recovery, Y + infection)


# Driver functions for the simulation, should be top level container in GrFN
def simulate():
    prams = (0.1, 0.05, 0.01, 1000.0, 0.1)
    tf = 200
    t = np.arange(0, tf, 0.1)
    tl = len(t)

    S = np.zeros(tl)
    I = np.zeros(tl)
    R = np.zeros(tl)
    Y = np.zeros(tl)

    u0 = (999, 1, 0, 0)
    S[0] = u0[0]
    I[0] = u0[1]
    R[0] = u0[2]
    Y[0] = u0[3]

    u = u0
    for i in range(1, tl):
        u = sir(u, prams)
        S[i] = u[0]
        I[i] = u[1]
        R[i] = u[2]
        Y[i] = u[3]

    return (t, S, I, R, Y)


seed(42)

# NOTE: printing output for potential correctness verification
(_, S, I, R, Y) = simulate()
print(S)
print(I)
print(R)
print(Y)
