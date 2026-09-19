# Rigid-rod homopolymer melt, SCFT, 3D — discrete worm-like-chain (WLC) statistics.
#
# Builds a stiff bead-rod homopolymer (`SCFTWormLikeChainFluid`/`WLCPropagator`, see the
# WLC tutorial in the docs) and converges it to its uniform bulk density (a single-species
# melt with no chi has no ordering transition — the density profile itself is featureless,
# exactly like the equivalent discrete-Gaussian-chain (DGC) melt would be). What differs
# between the two chain models is the underlying SINGLE-CHAIN statistics: a DGC chain of
# N beads/bond length b has <R^2> = (N-1)*b^2 regardless of stiffness, while a WLC chain
# with persistence length lp is far more extended for the same N,b — this script prints
# both predictions (via the exact discrete freely-rotating-chain formula,
# `ClassicalDFT.bending_eigenvalues`) so the difference is visible even though the SCFT
# density profile itself looks identical for both models at this bulk-melt condition.
using ClassicalDFT, CairoMakie

# --- Molecule ------------------------------------------------------------------------
N = 10           # beads per chain
b = 1.0          # bond length
lp = 8.0         # persistence length (stiff: lp/b = 8, well above N-1 = 9 bonds' own length)
rho0 = 1.0
kappa_helfand = 20.0

model = ClassicalDFT.SCFTWormLikeChainFluid([("rod", ["A"=>N])], [b], [lp], zeros(1,1);
                                             rho0=rho0, kappa=kappa_helfand)
mol_structure = Dict("rod" => ClassicalDFT.custom_structure("A"^N))

# --- Single-chain statistics (independent of the SCFT density profile) ---------------
# Exact discrete freely-rotating-chain formula for n=N-1 bonds, correlation
# c1 = <cos(bond angle)> = bending_eigenvalues(kappa,L_max)[2] (kappa = lp/b).
function frc_R2(b, κ, N, L_max=20)
    n = N - 1
    c1 = ClassicalDFT.bending_eigenvalues(κ, L_max)[2]
    return n*b^2*(1+c1)/(1-c1) - 2*b^2*c1*(1-c1^n)/(1-c1)^2
end
R2_wlc = frc_R2(b, lp/b, N)
R2_dgc = (N-1)*b^2   # the discrete-Gaussian-chain (ideal random walk) prediction
println("Single-chain <R^2>: WLC (lp/b=$(lp/b)) = ", round(R2_wlc, digits=2),
        "   vs DGC (same N,b) = ", round(R2_dgc, digits=2),
        "   ratio = ", round(R2_wlc/R2_dgc, digits=2))

# --- 3D structure and system -----------------------------------------------------------
# Orientation is inherently 3D, so WLCPropagator requires dimension(structure) == 3 (v1
# restriction — see WLCPropagator's docstring). Box size set generously relative to
# sqrt(R2_wlc) so periodic wraparound doesn't distort the (here, uniform anyway) profile.
L = 4 * sqrt(R2_wlc)
ngrid = (24, 24, 24)
structure = ClassicalDFT.Uniform3DCart((0.0, 0.0), [rho0], [0.0 L; 0.0 L; 0.0 L], ngrid)

n_chains = (L^3 * rho0) / N
system = ClassicalDFT.SCFTSystem(model, structure, ClassicalDFT.DFTOptions();
                          mol_structure=mol_structure, ensemble=[:canonical], n_molecules=[n_chains])

# --- Converge -----------------------------------------------------------------------
ρ = ClassicalDFT.initialize_profiles(system)
ClassicalDFT.converge!(system, ρ; verbose=true, log_interval=20, tol=1e-6, maxit=200)
ρtot = dropdims(sum(ρ, dims=4); dims=4)

println("incompressibility: ρ_total ∈ [", round(minimum(ρtot), digits=4), ", ",
        round(maximum(ρtot), digits=4), "]  (target ρ₀ = ", model.rho0, ")")

# --- Plot -----------------------------------------------------------------------
# A single-species bulk melt's density profile is trivially uniform (no chi to drive
# segregation) -- shown here as a sanity check on the converged profile, not because a
# flat heatmap is informative on its own; the real WLC-vs-DGC comparison is the printed
# <R^2> ratio above.
fig = Figure(size=(500, 400))
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="ρ_A(x,y,z=L/2)  (rigid-rod WLC melt)")
mid = ngrid[3] ÷ 2 + 1
hm = heatmap!(ax, ρ[:, :, mid, 1])
Colorbar(fig[1, 2], hm)
save("rigid_rod_wlc_melt.png", fig)
