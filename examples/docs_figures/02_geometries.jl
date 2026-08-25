# Figures for docs/src/tutorials/geometries.md
include("common.jl")
using Clapeyron, ClassicalDFT, CairoMakie

model = PCSAFT(["ethane"])
T, p = 298.0, 1e7
v = Clapeyron.volume(model, p, T, [1.0]; phase=:liquid)
ρbulk = [1/v]
L = ClassicalDFT.length_scale(model)
width = 5L

surface = Steele(["graphite"], width)

# Planar: fluid next to a single wall.
structure_planar = Uniform1DCart((p, T), ρbulk, [0.5L, width-0.5L], 201)
system_planar = DFTSystem(model, structure_planar, surface)
ρ_planar = ClassicalDFT.initialize_profiles(system_planar)
converge!(system_planar, ρ_planar)
save(assetpath("geometries_planar.png"), plot(system_planar, ρ_planar))

# Cylindrical: fluid inside a cylindrical pore of the same characteristic size.
structure_cyl = Uniform1DCyl((p, T), ρbulk, [0.0, width/2-0.5L], 201)
system_cyl = DFTSystem(model, structure_cyl, surface)
ρ_cyl = ClassicalDFT.initialize_profiles(system_cyl)
converge!(system_cyl, ρ_cyl)
save(assetpath("geometries_cylindrical.png"), plot(system_cyl, ρ_cyl))
