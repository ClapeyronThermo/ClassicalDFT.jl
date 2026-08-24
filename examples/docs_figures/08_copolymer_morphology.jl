# Figures for docs/src/tutorials/copolymer_morphology.md
#
# NOTE: this script has not been run/verified end-to-end in this environment (no Julia
# execution available) - please run it yourself before trusting the tutorial, and report
# back if `HeterogcPCPSAFT`'s tuple/userlocations constructor or the `mol_structure`
# DFTSystem kwarg need adjusting.
include("common.jl")
using Clapeyron, ClassicalDFT, CairoMakie

# Synthetic diblock "mol" = A4-B4, using the existing copolymer_params_*.csv (already in
# examples/, originally written for this same A/B synthetic pair).
model = HeterogcPCPSAFT([("mol", ["A"=>4, "B"=>4], [("A","A")=>3,("B","B")=>3,("A","B")=>1])];
                         userlocations=[joinpath(@__DIR__,"..","copolymer_params_like.csv"),
                                        joinpath(@__DIR__,"..","copolymer_params_unlike.csv")])

T, p = 298.15, 1e5
z = [1.0]
v = volume(model, p, T, z)
ρb = z ./ v .* 0.05   # dilute-ish average density, matching dynamic_dft_3d_copolymer.jl

L = ClassicalDFT.length_scale(model)
ngrid = 31
mol_structure = Dict("mol" => custom_structure("AAAABBBB"))

function converged_profile(structure)
    system = DFTSystem(model, structure; mol_structure=mol_structure)
    ρ = ClassicalDFT.initialize_profiles(system)
    converge!(system, ρ)
    return system, ρ
end

# Lamellar
structure_lam = ClassicalDFT.LamellarStack3DCart((p,T), ρb, [-10L 10L; -10L 10L; -10L 10L], (ngrid,ngrid,ngrid); core_groups=["A"])
system_lam, ρ_lam = converged_profile(structure_lam)
save(assetpath("copolymer_morphology_lamellar.png"), plot(system_lam, ρ_lam; plot_by=:group, color_by=:group))

# Hexagonally-packed cylinders (rectangular 2-cylinder supercell: Ly ≈ √3 Lx)
Lx = 10L
bounds_hex = [-Lx Lx; -sqrt(3)*Lx sqrt(3)*Lx; -Lx Lx]
ngrid_hex = (ngrid, round(Int, ngrid*sqrt(3)), ngrid)
structure_hex = ClassicalDFT.HexLattice3DCart((p,T), ρb, bounds_hex, ngrid_hex; core_groups=["A"])
system_hex, ρ_hex = converged_profile(structure_hex)
save(assetpath("copolymer_morphology_hex.png"), plot(system_hex, ρ_hex; plot_by=:group, color_by=:group))

# BCC spheres (cubic unit cell)
structure_bcc = ClassicalDFT.BCC3DCart((p,T), ρb, [-10L 10L; -10L 10L; -10L 10L], (ngrid,ngrid,ngrid); core_groups=["A"])
system_bcc, ρ_bcc = converged_profile(structure_bcc)
save(assetpath("copolymer_morphology_bcc.png"), plot(system_bcc, ρ_bcc; plot_by=:group, color_by=:group))

# Gyroid (cubic unit cell)
structure_gyr = ClassicalDFT.Gyroid3DCart((p,T), ρb, [-10L 10L; -10L 10L; -10L 10L], (ngrid,ngrid,ngrid); core_groups=["A"])
system_gyr, ρ_gyr = converged_profile(structure_gyr)
save(assetpath("copolymer_morphology_gyroid.png"), plot(system_gyr, ρ_gyr; plot_by=:group, color_by=:group))

println("saved copolymer_morphology_{lamellar,hex,bcc,gyroid}.png to ", ASSETS)
