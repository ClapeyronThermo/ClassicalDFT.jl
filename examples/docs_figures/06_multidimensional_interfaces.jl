# Figure for docs/src/tutorials/multidimensional_interfaces.md
include("common.jl")
using Clapeyron, ClassicalDFT, CairoMakie

model = PCSAFT(["water"])
T = 298.15
(p, vl, vv) = saturation_pressure(model, T)
ρl, ρv = [1.0]./vl, [1.0]./vv
L = ClassicalDFT.length_scale(model)

ngrid = 51
structure = ClassicalDFT.TwoPhase3DSphrCart((p, T), ρl, ρv, [-10L 10L; -10L 10L; -10L 10L], (ngrid, ngrid, ngrid))
system = DFTSystem(model, structure)
ρ = ClassicalDFT.initialize_profiles(system)
converge!(system, ρ)

fig = plot(system, ρ)
save(assetpath("multidimensional_interfaces_droplet.png"), fig)
println("saved ", assetpath("multidimensional_interfaces_droplet.png"))
