# Figures for docs/src/tutorials/planar_adsorption.md
include("common.jl")
using Clapeyron, ClassicalDFT, CairoMakie

model = PCSAFT(["carbon dioxide"])
p, T = 1e6, 298.15
L = ClassicalDFT.length_scale(model)

# Slit-pore density profile.
width = 50e-10
surface = Steele(["graphite"], width)
v = Clapeyron.volume(model, p, T, [1.0])
ρbulk = [1/v]
structure = Uniform1DCart((p, T), ρbulk, [0.6L, width - 0.6L], 201)
system = DFTSystem(model, structure, surface)
ρ = ClassicalDFT.initialize_profiles(system)
converge!(system, ρ)
save(assetpath("planar_adsorption_profile.png"), plot(system, ρ))

# Adsorption isotherm vs. pore width.
widths = range(15e-10, 60e-10, length=20)
ads = [adsorption(model, Steele(["graphite"], w), p, T)[1] for w in widths]

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="pore width / Å", ylabel="adsorption / (mol/m³)")
lines!(ax, widths .* 1e10, ads; linewidth=3)
scatter!(ax, widths .* 1e10, ads)
save(assetpath("planar_adsorption_isotherm.png"), fig)

println("saved planar_adsorption_{profile,isotherm}.png to ", ASSETS)
