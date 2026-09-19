# AB diblock copolymer, SCFT, 1D lamellar — discrete worm-like-chain (WLC) statistics
# with DIFFERENT persistence lengths per block, plus a visualization of the local
# preferred bond orientation.
#
# `WLCPropagator` supports `dimension(structure) ∈ {1,2,3}` (see its docstring): a 1D
# structure like this one tracks density along `x` only, treating the system as
# translationally invariant in `y,z`, while bond orientation is still tracked in full 3D
# (bending correlations are inherently 3D) — so a 1D system is exactly where "preferred
# orientation" is easiest to visualize: `orientation_order_parameter` reports how bonds
# align relative to the SAME `x` axis the density profile itself varies along.
#
# `A` is stiff (`lp_A=5`, comparable to its own contour length) and `B` is flexible
# (`lp_B=0.5`) — v1 requires every bond in a `SCFTWormLikeChainFluid` chain to be a
# bead-rod bond (no mixing with `DiscreteGaussianChainPropagator` bonds), but nothing
# stops two WLC species in the SAME model from having very different persistence
# lengths, which is exactly what's needed here.
using ClassicalDFT

# --- Molecule and interactions ---------------------------------------------------
N_seg = 20
N_A, N_B = N_seg ÷ 2, N_seg ÷ 2
chi_val = 1.5
chi = zeros(2, 2)
chi[1, 2] = chi[2, 1] = chi_val

b = [1.0, 1.0]
lp = [5.0, 0.5]   # A: stiff (lp/b = 5); B: flexible (lp/b = 0.5)

# L_max=6 (vs. the default 8) trades some accuracy in resolving A's bending kernel for
# roughly 40% fewer orientation-grid nodes (n_orient = (L_max+1)(2L_max+1)) — worthwhile
# here since bending_eigenvalues(5.0, 6)[end] ≈ 0.017, already small. Every SCFT
# iteration costs noticeably more than the equivalent DiscreteGaussianChainPropagator
# system (the propagator does an FFT convolution per orientation node per bond, not
# just one per bond) — this example takes several minutes to converge, not seconds.
model = ClassicalDFT.SCFTWormLikeChainFluid([("diblock", ["A"=>N_A, "B"=>N_B])], b, lp, chi;
                                             rho0=1.0, kappa=25.0, L_max=6)
mol_structure = Dict("diblock" => ClassicalDFT.custom_structure("A"^N_A * "B"^N_B))

# --- 1D structure and system ---------------------------------------------------------
# LamellarStack1DCart seeds a periodic layered profile directly, a much more reliable
# starting point than random noise (see the 1D diblock tutorial, docs/src/tutorials/scft.md).
L = 7.0
ngrid = 33
structure = ClassicalDFT.LamellarStack1DCart((0.0, 0.0), [1.0], [0.0, L], ngrid; core_groups=["A"])

n_chains = L / N_seg
system = ClassicalDFT.SCFTSystem(model, structure, ClassicalDFT.DFTOptions();
                          mol_structure=mol_structure, ensemble=[:canonical], n_molecules=[n_chains])

# --- Converge -----------------------------------------------------------------------
ρ = ClassicalDFT.initialize_profiles(system)
ClassicalDFT.converge!(system, ρ; verbose=true, log_interval=10, tol=1e-5, maxit=300)

ρtot = sum(ρ, dims=2)
println("incompressibility: ρ_total ∈ [", round(minimum(ρtot), digits=3), ", ",
        round(maximum(ρtot), digits=3), "]  (target ρ₀ = ", model.rho0, ")")

# --- Local preferred orientation ------------------------------------------------------
# orientation_order_parameter needs q_in/q_out resolved in orientation (not just ρ, which
# is already marginalized over u), so it's computed from one more propagate! call at the
# converged field — S_α(x) = (3⟨u_x²⟩_α(x)-1)/2: S=1 bonds aligned along x (the domain
# normal), S=-1/2 bonds confined to the y-z plane (perpendicular to the domain normal),
# S=0 isotropic (what a DiscreteGaussianChainPropagator chain always has — it has no
# orientation to align in the first place, so this diagnostic only exists for WLC chains).
w = zeros(size(ρ)...)
w_bulk = ClassicalDFT.compute_bulk_fields(system.model, ClassicalDFT.compute_bulk_densities(system))
ClassicalDFT.compute_fields!(system, ρ, w)
cache_propagator = ClassicalDFT.preallocate_propagator(system, system.propagator, ρ, CPU())
ClassicalDFT.propagate!(system, ρ, w, cache_propagator; w_bulk=w_bulk)
q_in = ClassicalDFT.cache_q_in(cache_propagator)
q_out = ClassicalDFT.cache_q_out(cache_propagator)
S = ClassicalDFT.orientation_order_parameter(system, q_in, q_out; axis=1)

println("S_A (stiff block)    range: ", round.(extrema(S[:, 1]); digits=3))
println("S_B (flexible block) range: ", round.(extrema(S[:, 2]); digits=3))

# --- Plot -----------------------------------------------------------------------
# Two panels sharing the x-axis: volume fractions (top) and the orientation order
# parameter per species (bottom) — the interesting result is that S_A stays strongly
# negative (bonds lying flat, perpendicular to the domain normal) almost everywhere,
# most strongly where A is a dilute minority intruding into the B domain, while S_B
# turns slightly positive in the B domain's interior (a mild parallel/stretching
# tendency) and only becomes strongly negative where B is itself a dilute minority.
using CairoMakie
x = range(0, L, ngrid)
fig = Figure(size=(650, 550))
ax1 = Axis(fig[1, 1], ylabel="volume fraction φ", title="Diblock WLC melt (A stiff, B flexible): density and preferred orientation")
lines!(ax1, x, ρ[:, 1] ./ vec(ρtot), label="φ_A (stiff, lp=5)")
lines!(ax1, x, ρ[:, 2] ./ vec(ρtot), label="φ_B (flexible, lp=0.5)")
axislegend(ax1)
ax2 = Axis(fig[2, 1], xlabel="x", ylabel="order parameter S(x)")
lines!(ax2, x, S[:, 1], label="S_A")
lines!(ax2, x, S[:, 2], label="S_B")
hlines!(ax2, [0.0]; color=:gray, linestyle=:dash)
axislegend(ax2)
save("diblock_differential_stiffness_wlc.png", fig)
