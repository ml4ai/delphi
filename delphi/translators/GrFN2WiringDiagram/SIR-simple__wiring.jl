using Catlab
using Catlab.WiringDiagrams
using Catlab.Doctrines
import Catlab.Doctrines.⊗
import Base: ∘

include("SIR-simple__functions.jl")
⊗(a::WiringDiagram, b::WiringDiagram) = otimes(a, b)
∘(a::WiringDiagram, b::WiringDiagram) = compose(b, a)
⊚(a,b) = b ∘ a

β, S0, I0, R0, δt, γ = Ob(FreeSymmetricMonoidalCategory, :β, :S0, :I0, :R0, :δt, :γ)
rec, S1, I1, R1, inf = Ob(FreeSymmetricMonoidalCategory, :rec, :S1, :I1, :R1, :inf)

w0_0 = WiringDiagram(Hom(ID, S0, S0))
w0_1 = WiringDiagram(Hom(ID, I0, I0))
w0_2 = WiringDiagram(Hom(ID, R0, R0))

w1_0 = WiringDiagram(Hom(assg__rec, γ ⊗ I0 ⊗ δt, rec))
w1_1 = WiringDiagram(Hom(assg__inf, β ⊗ S0 ⊗ I0 ⊗ R0 ⊗ δt, inf))

w2_0 = WiringDiagram(Hom(assg__S1, S0 ⊗ inf, S1))
w2_1 = WiringDiagram(Hom(assg__I1, I0 ⊗ inf ⊗ rec, I1))
w2_2 = WiringDiagram(Hom(assg__R1, R0 ⊗ rec, R1))

w3_0 = (w0_0 ⊗ w1_1) ⊚ w2_0
w3_1 = (w0_1 ⊗ w1_1 ⊗ w1_0) ⊚ w2_1
w3_2 = (w0_2 ⊗ w1_0) ⊚ w2_2

w4_0 = w3_0 ⊗ w3_1 ⊗ w3_2
println(w4_0)
