# Figure for docs/src/tutorials/vapor_liquid_interfaces.md
include("common.jl")
using Clapeyron, ClassicalDFT, CairoMakie

model = PCSAFT(["water"])
T = 298.15
(p, vl, vv) = saturation_pressure(model, T)
ρ1, ρ2 = [1.0]./vl, [1.0]./vv
L = ClassicalDFT.length_scale(model)

structure = TwoPhase1DCart((p, T), ρ1, ρ2, [-10L, 10L], 201)
system = DFTSystem(model, structure)
ρ = ClassicalDFT.initialize_profiles(system)
converge!(system, ρ)

fig = plot(system, ρ)
save(assetpath("vapor_liquid_interfaces_profile.png"), fig)
println("saved ", assetpath("vapor_liquid_interfaces_profile.png"))
