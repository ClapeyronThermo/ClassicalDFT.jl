# Figure for docs/src/tutorials/group_contribution_chains.md
# Needs GCIdentifier + ChemicalIdentifiers loaded for HeterogcPCPSAFT(["1-butanol"])'s
# automatic name -> connectivity resolution.
include("common.jl")
using Clapeyron, ClassicalDFT, GCIdentifier, ChemicalIdentifiers, CairoMakie

model = HeterogcPCPSAFT(["1-butanol"])
T = 298.15
(p, vl, _) = Clapeyron.saturation_pressure(model, T)
ρbulk = [1/vl]
L = ClassicalDFT.length_scale(model)

width = 6L

surface = Steele(["graphite"], width)
structure = Uniform1DCart((p, T), ρbulk, [0.5L, width-0.5L], 201)
system = DFTSystem(model, structure, surface)
ρ = ClassicalDFT.initialize_profiles(system)
converge!(system, ρ)

fig = plot(system, ρ; y_units=:mol)
save(assetpath("group_contribution_chains_profile.png"), fig)
println("saved ", assetpath("group_contribution_chains_profile.png"))
