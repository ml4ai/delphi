using Catlab
using Catlab.WiringDiagrams
using Catlab.Doctrines
import Catlab.Doctrines: ⊗, id
import Base: ∘

include("SIR-simple__functions.jl")
⊗(a::WiringDiagram, b::WiringDiagram) = otimes(a, b)
∘(a::WiringDiagram, b::WiringDiagram) = compose(b, a)
⊚(a,b) = b ∘ a

# ==============================================================================
# Translated wiring of the SIR-simple model
# from GrFN to WiringDiagram specification
# ==============================================================================
β, S0, I0, R0, δt, γ = Ob(FreeSymmetricMonoidalCategory, :β, :S0, :I0, :R0, :δt, :γ)
rec, inf = Ob(FreeSymmetricMonoidalCategory, :rec, :inf)
S1, I1, R1 = Ob(FreeSymmetricMonoidalCategory, :S1, :I1, :R1)

id_S0 = id(Ports([S0]))
id_I0 = id(Ports([I0]))
id_R0 = id(Ports([R0]))

IN_0 = WiringDiagram(Hom(:L0_REWIRE, S0 ⊗ I0 ⊗ R0 ⊗ β ⊗ γ ⊗ δt, β ⊗ S0 ⊗ I0 ⊗ R0 ⊗ δt ⊗ S0 ⊗ I0 ⊗ R0 ⊗ γ ⊗ I0 ⊗ δt))
WD_inf = WiringDiagram(Hom(assg__inf, β ⊗ S0 ⊗ I0 ⊗ R0 ⊗ δt, inf))
WD_rec = WiringDiagram(Hom(assg__rec, γ ⊗ I0 ⊗ δt, rec))
OUT_1 = IN_0 ⊚ (WD_inf ⊗ id_S0 ⊗ id_I0 ⊗ id_R0 ⊗ WD_rec)

IN_1 = WiringDiagram(Hom(:L1_REWIRE, inf ⊗ S0 ⊗ I0 ⊗ R0 ⊗ rec, S0 ⊗ inf ⊗ I0 ⊗ inf ⊗ rec ⊗ R0 ⊗ rec))
WD_S1 = WiringDiagram(Hom(assg__S1, S0 ⊗ inf, S1))
WD_I1 = WiringDiagram(Hom(assg__I1, I0 ⊗ inf ⊗ rec, I1))
WD_R1 = WiringDiagram(Hom(assg__R1, R0 ⊗ rec, R1))
OUT_2 = OUT_1 ⊚ IN_1 ⊚ (WD_S1 ⊗ WD_I1 ⊗ WD_R1)


println(OUT_2)
