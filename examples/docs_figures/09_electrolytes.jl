# Figure for docs/src/tutorials/electrolytes.md
include("common.jl")
using Clapeyron, ClassicalDFT, CairoMakie

model = ePCSAFT(["water08"], ["sodium", "chloride"])
model.neutralmodel.params.epsilon.values[2,2] = 197.737^2/70.0
model.neutralmodel.params.epsilon.values[3,3] = 70.0
T, p = 298.15, 1e7
x = [0.9, 0.05, 0.05]
v = Clapeyron.volume(model.neutralmodel, p, T, x)
ρbulk = x ./ v
L = ClassicalDFT.length_scale(model.neutralmodel)
width = 5L
surface = Steele(["graphite"], width)

structure = Uniform1DCart((p, T), ρbulk, [0.5L, width-0.5L], 201)
system = ClassicalDFT.ElectrolyteDFTSystem(model, structure, surface)
ρ = initialize_profiles(system)
converge!(system, ρ)

fig = plot(system, ρ)
save(assetpath("electrolytes_profile.png"), fig)
println("saved ", assetpath("electrolytes_profile.png"))
