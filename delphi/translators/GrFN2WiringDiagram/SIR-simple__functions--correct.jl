function assg__rec(γ, I, δt)
    return (γ*I) * δt
end

function assg__inf(β, S, I, R, δt)
    return (-(β*S*I) / (S+I+R)) * δt
end

function assg__I1(I, infected, recovered)
    return I + infected - recovered
end

function assg__R1(R, recovered)
    return R + recovered
end

function assg__S1(S, infected)
    return S - infected
end

function ID(x)
    return x
end
