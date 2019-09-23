function SIR_simple__sir__assign__I__0(I, infected, recovered)
    return I + infected - recovered
end

function SIR_simple__sir__assign__S__0(S, infected)
    return S - infected
end

function SIR_simple__sir__assign__R__0(R, recovered)
    return R + recovered
end

function SIR_simple__sir__assign__infected__0(beta, S, I, R, dt)
    return (-(beta*S*I) / (S + I + R)) * dt
end

function SIR_simple__sir__assign__recovered__0(gamma, I, dt)
    return (gamma*I) * dt
end