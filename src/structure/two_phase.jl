function __coeff_cos_prof_correlation(model,T,scale = one(T))
    Tc,_,_ = crit_pure(model)
    Tr = T/Tc
    c0 = 2.4728 - 2.3625*Tr
    return c0 * scale
end

"""
    initialize_profiles(model::ElectrolyteModel, structure::DFTStructure{1,Cartesian,TwoPhaseSystem{:Cartesian}}, species, device, FP)

`ElectrolyteModel`-specific override of the generic `cos_prof`-based method
below: builds a periodic double-interface ("slab") tanh profile instead, one
shape per Clapeyron *component* (`model.groups.i_groups[c]`, from that
component's own `structure.ρbulk[c]`/`structure.topology.ρbulk2[c]`) rather
than one shared shape derived from `Clapeyron.split_pure_model`.

The generic method tunes `cos_prof`'s cosmetic steepness via
`crit_pure(Clapeyron.split_pure_model(model)[i])` for each component --
`split_pure_model` requires an `is_splittable`/`default_splitter` method for
every component, which a bare free-ion component (no meaningful "pure
critical point" of its own -- an isolated ion doesn't phase-separate) does
not generally have, and even where a fallback exists, that ion's own
isolated-pure-substance critical behavior has no bearing on the *mixture*'s
actual interfacial width. A simple, fixed-width tanh profile sidesteps this
entirely, matching the periodicity `evaluate_field!`/`propagate!`'s FFT-based
machinery requires (a *discontinuous* one-sided profile gets smoothed away
within the first few SCF iterations into a spurious uniform intermediate
density, reported as a clean converged residual -- a silent failure mode).

For a single-component neutral system, every bead shares one tanh shape
(same as `cos_prof` would give up to cosmetic steepness); for a genuine
multi-component (chain + ion(s)) electrolyte, each component's own bulk
densities (generally in a different ratio between phases than the chain's)
get their own shape. Two corrections are applied automatically afterward by
the generic `initialize_profiles(system::AbstractcDFTSystem)` entry point --
calling this method directly gives neither: (1) the overall
(domain-integrated) electroneutrality Boltzmann rescale (`find_ψ_const`), a
single global constant correcting the box's TOTAL charge to zero, and (2),
when the two bulk phases sit at different electrochemical (Donnan)
potentials (`Ψ≠0`, via [`impose_donnan_structure!`](@ref)), a local
charge-layer correction seeding the correct interfacial structure a genuine
potential jump requires -- see that function's docstring.
"""
function initialize_profiles(model::ElectrolyteModel, structure::DFTStructure{1,Cartesian,TwoPhaseSystem{:Cartesian}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    width_frac = FP(0.05)
    lb, ub = bounds(structure, 1)
    H = ub - lb
    x = collect(uniform_range(structure, 1))
    s = @. (x - lb) / H

    nbeads = sum(species.nbeads)
    ngrid = structure.ngrid
    ρ = allocate(device, FP, ngrid..., nbeads)

    for c in 1:length(model)
        ρ1 = structure.ρbulk[c]           # phase 1 -- middle half of the box
        ρ2 = structure.topology.ρbulk2[c] # phase 2 -- outer quarters
        ρ_points = @. ρ2 + (ρ1 - ρ2) * 0.5 * (tanh((s - 0.25) / width_frac) - tanh((s - 0.75) / width_frac))
        ρ_points_dev = adapt_to_device(device, FP, ρ_points)
        for j in model.groups.i_groups[c]
            ρ[:, j] = ρ_points_dev
        end
    end
    return ρ
end

"""
    impose_donnan_structure!(structure::DFTStructure{1,Cartesian,TwoPhaseSystem{:Cartesian}}, ef::ElectrostaticPotentialModel, model::ElectrolyteModel, ρ)

Rescales the naive, uncorrected two-phase tanh IC (built just above) so it
starts from a profile that already carries the *correct* local
charge-separation structure supporting the two bulk phases' actual
electrochemical (Donnan) potential difference `Ψ`, rather than the
exactly-zero-charge, exactly-zero-potential-jump state the naive tanh guess
produces everywhere by construction (every bead of one component shares an
identical transition shape, so `Σ_c Z_c·ρ_c(s)` collapses to a difference of
two individually-electroneutral bulk phases -- identically zero for every
`s`). At the true two-phase fixed point there MUST instead be a nonzero
local charge-density layer at each interface (Poisson's equation: a
spatially-varying potential requires nonzero `∇²ψ` there); starting the SCF
iteration from the exactly-flat, wrong-topology naive guess instead of this
structure is a known failure mode for genuinely `Ψ≠0` systems (confirmed:
every mixing-scheme/IC-orientation variant tried instead converges to an
unrelated, exact-but-wrong homogeneous fixed point of the same map). Seeding
the correct electrostatic structure directly is an initial-guess change, not
a solver change.

`Ψ` itself is read off directly from the two ALREADY-CONVERGED bulk phases
already stored on `structure` (`Clapeyron.donnan_psi_bulk`, no extra solve)
-- fully automatic, no caller-supplied potential value needed. Returns `ρ`
untouched (and does no work) whenever `model` has no charged components at
all, or its bulk phases happen to be Donnan-neutral (`Ψ≈0` to machine
precision, e.g. a net-neutral chain + symmetric salt, whose cation↔anion
charge-swap symmetry forces `Ψ≡0` exactly) -- the naive IC is already exact
in that case.

**Method**: build the SAME prescribed tanh potential profile `ψ0(z)` used
for density (same interface locations/width, going from `0` to `Ψ`, reduced
`k_BT/e` units), get the local charge density `q(z) = -ψ0''(z)/_c` Poisson's
equation says `ψ0(z)` requires (`_c = N_A·e_c²/(ε₀·ϵ_r)`, the SAME constant
`ElectrostaticPotential` builds into its own Fourier kernel, read directly
off `ef.ϵ_r`, computed here analytically via `ψ0`'s closed-form second
derivative -- no FFT needed), then solve, AT EVERY GRID POINT `z`
simultaneously (vectorized over the grid, not a per-point scalar loop, so
this stays GPU-safe), for a per-point scalar `η(z)` such that

    Σ_c Z_c·ρ_old_c(z)·exp(-Z_c·η(z)) = q(z)

(damped Newton, `Δ=clamp(...,-1,1)` per iteration, `η` itself clamped to
`±15` -- both safety nets against the dilute plateau's vanishing
sensitivity `dq/dη ∝ ρ_c` forcing an outsized raw step/target), then rescale
every bead of component `c` uniformly by `ρ_c(z) ← ρ_c(z)·exp(-Z_c·η(z))`
(uniform across that component's beads, consistent with them already
sharing one profile; required, not just a convention match, for the same
reason a per-bead correction would be wrong -- see above).

The MULTIPLICATIVE Boltzmann form is used here, not the mathematically
simpler closed-form additive correction the *linearized* (small-`η`) limit
of this same equation would give (`ρ_new_c ≈ ρ_old_c + [Z_cρ_old_c/Σ_dZ_d²ρ_old_d]·q(z)`,
exact only to first order): that additive form has NO positivity guarantee
for a genuinely large `Ψ` (confirmed directly: for `Ψ≈-3`, ~7% of the grid
went density-negative, exactly where `Σ_dZ_d²ρ_old_d(z)` is locally small
relative to `q(z)`) and is a real bug, not a cosmetic one, once it happens.
`ρ_c·exp(-Z_cη)` stays strictly positive for any finite `η` by construction,
regardless of how large `Ψ` is.
"""
function impose_donnan_structure!(structure::DFTStructure{1,Cartesian,TwoPhaseSystem{:Cartesian}}, ef::ElectrostaticPotentialModel, model::ElectrolyteModel, ρ)
    Z = Clapeyron.component_charges(model)
    all(iszero, Z) && return ρ

    T = structure.conditions[2]
    Ψ = Clapeyron.donnan_psi_bulk(model, T, structure.ρbulk, structure.topology.ρbulk2)
    isapprox(Ψ, 0.0; atol=1e-10) && return ρ

    width_frac = 0.05
    lb, ub = bounds(structure, 1)
    H = ub - lb
    x = collect(uniform_range(structure, 1))
    s = @. (x - lb) / H

    u1 = @. (s - 0.25) / width_frac
    u2 = @. (s - 0.75) / width_frac
    d2ψ0_ds2 = @. Ψ / width_frac^2 * (-sech(u1)^2 * tanh(u1) + sech(u2)^2 * tanh(u2))
    ψ0_phys_pp = d2ψ0_ds2 ./ H^2 .* (k_B * T)  # d²ψ0/dx², physical units

    _c = N_A * e_c^2 / ϵ_0 / ef.ϵ_r
    q_target = @. -ψ0_phys_pp / _c   # (ngrid,)

    N = length(model)
    rep_bead = [model.groups.i_groups[c][1] for c in 1:N]

    Zc = reshape(Z, 1, N)
    ρc = reduce(hcat, (ρ[:, rep_bead[c]] for c in 1:N))  # ngrid × N

    η = zero(q_target)  # (ngrid,)
    for _ in 1:200
        expfac = @. exp(-η * Zc)                          # ngrid × N
        q = vec(sum(ρc .* expfac .* Zc, dims=2))          # ngrid
        dq = vec(sum(.-ρc .* expfac .* Zc .^ 2, dims=2))  # ngrid
        Δ = clamp.((q .- q_target) ./ dq, -1.0, 1.0)
        η = clamp.(η .- Δ, -15.0, 15.0)
        maximum(abs.(q .- q_target)) < 1e-10 * max(maximum(abs.(q_target)), 1.0) && break
    end

    factor = @. exp(-η * Zc)  # ngrid × N
    for c in 1:N
        fac_c = @view factor[:, c]
        for k in model.groups.i_groups[c]
            ρ[:, k] .*= fac_c
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{1,Cartesian,TwoPhaseSystem{:Cartesian}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub-lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure, 1) |> collect
    X = collect(x)

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,H/L)
        coef = sqrt(coef^2-1)/4
        for j in @chain(i)
            ρ_points = @. cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ[:,j] = adapt_to_device(device, FP, ρ_points)
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{2,Cartesian,TwoPhaseSystem{:Lamellar}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure, 1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:] .= x[i]
    end

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,H/L)
        coef = sqrt(coef^2-1)/4
        for j in @chain(i)
            ρ_points = @.  cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,3,j) .= ρ_points
        end
    end
    return ρ
end


function initialize_profiles(model::EoSModel,structure::DFTStructure{3,Cartesian,TwoPhaseSystem{:Lamellar}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,H/L)
        coef = sqrt(coef^2-1)/4

        for j in @chain(i)
            ρ_points = @.  cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,4,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{2,Cartesian,TwoPhaseSystem{:HexLattice}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:] .= x[i]
    end

    y = uniform_range(structure,2)
    Y = zeros(ngrid)

    for i in 1:ngrid[2]
        Y[:,i] .= y[i]
    end

    r = sqrt.(X.^2 + Y.^2)
  
    L = length_scale(model)
    R = H/sqrt(2π)

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,1/L)
        for j in @chain(i)
            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],R,coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,3,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{3,Cartesian,TwoPhaseSystem{:HexLattice}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end

    y = uniform_range(structure,2)
    Y = zeros(ngrid)

    for i in 1:ngrid[2]
        Y[:,i,:] .= y[i]
    end

    r = sqrt.(X.^2 + Y.^2)

    L = length_scale(model)
    R = H/sqrt(2π)

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,1/L)
        for j in @chain(i)
            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],R,coef)
            selectdim(ρ,4,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{3,Cartesian,TwoPhaseSystem{:Spherical}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end

    y = uniform_range(structure,2)
    Y = zeros(ngrid)

    for i in 1:ngrid[2]
        Y[:,i,:] .= y[i]
    end

    z = uniform_range(structure,3)
    Z = zeros(ngrid)

    for i in 1:ngrid[3]
        Z[:,:,i] .= z[i]
    end

    r = sqrt.(X.^2 + Y.^2 + Z.^2)

    L = length_scale(model)
    R = H*(3/(8π))^(1/3)

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,1/L)
        for j in @chain(i)
            ρ_points = @. tanh_prof(r, ρ1[i], ρ2[i], R, coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,4,j) .= ρ_points
        end
    end
    return ρ
end