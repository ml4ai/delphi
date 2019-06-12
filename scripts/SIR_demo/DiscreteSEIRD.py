# ==============================================================================
# NOTE: I am not really sure what is going on here because the GTRI provide no
# sample inputs in there version of discrete SEIRD.

# However what I think is happening is that they are returning deltas for each
# of the now 5 groups and then handling the computation with those deltas
# elsewhere. Not sure why `t` is present as it serves no purpose.
# ==============================================================================


def seird(du, u, parms, t):
    S, E, Ir, Id, R, D = u
    β, δ, γ, Γ, μ, ϵ, ω = parms
    I = Ir + Id
    N = S + E + I + R + D
    dS = S - β*S*I/N + ω*R
    dE = E + β*S*I/N - γ*E + ϵ
    dIr = Ir + γ*(1-μ)E - γ*Ir + ϵ
    dId = Id + γ*μ*E - Γ*Id + ϵ
    dR = R + γ*Ir - ω*R
    dD = D + Γ*Id
    return (dS, dE, dIr, dId, dR, dD)
