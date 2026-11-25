# Figures for docs/src/tutorials/scft.md (semiflexible and rigid chains section)
#
# A rod-coil diblock (A: exact rigid rod, lp=Inf; B: flexible coil), parameters chosen to
# match Tang et al. 2015 (Macromolecules 48, 9060)'s Fig. 11 lamellar point (chiN=16,
# nu^-2=10, f=0.7), with N=20 (a=b assumed equal Kuhn/bond convention; not pinned down by
# that figure specifically) and the domain period (L=16.5) found by minimizing free
# energy over the period beforehand -- not re-derived here, to keep the tutorial focused.
include("common.jl")
using ClassicalDFT, CairoMakie, Random

N_A, N_B = 14, 6
N_seg = N_A + N_B
f = N_A / N_seg
b = [1.0, sqrt(2)]
lp = [Inf, 0.5*sqrt(2)]
chi = zeros(2, 2); chi[1,2] = chi[2,1] = 0.8
L_max = 6
Lx, Ly = 16.5, 4.0
ngrid = (67, 17)

model = SCFTWormLikeChainFluid([("diblock", ["A"=>N_A, "B"=>N_B])], b, lp, chi;
                                rho0=1.0, kappa=25.0, L_max=L_max)
mol_structure = Dict("diblock" => custom_structure("A"^N_A * "B"^N_B))
structure = ClassicalDFT.LamellarStack2DCart((0.0, 0.0), [1.0], [0.0 Lx; 0.0 Ly], ngrid;
                                              core_groups=["A"], core_fraction=f)
n_chains = (Lx * Ly) / N_seg
system = SCFTSystem(model, structure, DFTOptions();
                     mol_structure=mol_structure, ensemble=[:canonical], n_molecules=[n_chains])

ρ = ClassicalDFT.initialize_profiles(system)
converge!(system, ρ; verbose=true, tol=1e-5, maxit=3000, beta=5e-3)

save(assetpath("wlc_density.png"), plot(system, ρ))

# Orientation and mean-orientation-field require q_in/q_out resolved in orientation (not
# just rho, which is already marginalized over u) -- plot_orientation_field/
# terminal_orientation_anchor/sample_chain all compute that internally from rho (one more
# propagate! call at the converged field, exactly as compute_densities! itself does
# internally) since q_in/q_out aren't passed in below. Computed once by hand here instead
# (compute_propagator_fields) purely to reuse across the anchor/sample_chain calls that
# follow, rather than repeating that propagate! call for each of them.
q_in, q_out = ClassicalDFT.compute_propagator_fields(system, ρ)

save(assetpath("wlc_orientation_field.png"),
     ClassicalDFT.plot_orientation_field(system, ρ; species=1, q_in=q_in, q_out=q_out))

r0, u0, mean_u = ClassicalDFT.terminal_orientation_anchor(system, ρ, 1; orientation=:mean, q_in=q_in, q_out=q_out)
R1 = ClassicalDFT.sample_chain(system, ρ, 1, r0, u0; rng=MersenneTwister(1), q_in=q_in, q_out=q_out)
r0N, u0N, _ = ClassicalDFT.terminal_orientation_anchor(system, ρ, N_seg; orientation=:mean, q_in=q_in, q_out=q_out)
R2 = ClassicalDFT.sample_chain(system, ρ, N_seg, r0N, u0N; rng=MersenneTwister(2), q_in=q_in, q_out=q_out)
save(assetpath("wlc_chain_conformations.png"),
     ClassicalDFT.plot_chain_conformations(system, ρ, [R1, R2]))

println("saved wlc_{density,orientation_field,chain_conformations}.png to ", ASSETS)
