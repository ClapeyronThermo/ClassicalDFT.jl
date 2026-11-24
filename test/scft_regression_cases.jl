# Frozen SCFT regression cases (Phase 0 of the modular-SCFT refactor).
#
# Each case builds a system through the *public* constructors only, runs a fixed number of
# `converge!` iterations from a deterministic seed, and reduces the resulting density to a
# few scalar observables. Golden values live in `test_scft_regression.jl`; regenerate them
# only deliberately, with `julia --project test/scft_regression_generate.jl`.
using ClassicalDFT

const _REG_OPTS = (verbose=false, tol=1e-10)

_seed_cos!(ρ, z, L, amp, sgn) = (ρ .+= sgn .* amp .* cos.(2π .* z ./ L); ρ)

function _reg_lamella_dgc()
    N, χ, L, ng = 20, 1.5, 7.0, 65
    model = SCFTLatticeFluid([("diblock", ["A"=>N÷2, "B"=>N÷2])], [1.0, 1.0], [0.0 χ; χ 0.0];
                             rho0=1.0, kappa=20.0)
    structure = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng)
    system = ClassicalDFT.SCFTSystem(model, structure, ClassicalDFT.DFTOptions();
        mol_structure=Dict("diblock" => ClassicalDFT.custom_structure("A"^(N÷2) * "B"^(N÷2))),
        ensemble=[:canonical], n_molecules=[L/N])
    ρ = ClassicalDFT.initialize_profiles(system)
    z = range(0, L, length=ng)
    _seed_cos!(view(ρ, :, 1), z, L, 0.05,  1)
    _seed_cos!(view(ρ, :, 2), z, L, 0.05, -1)
    ClassicalDFT.converge!(system, ρ; _REG_OPTS..., maxit=400)
    return ρ
end

function _reg_star_blend_gc()
    # Branched AB star (canonical) + grand-canonical A homopolymer solvent, χ>0.
    L, ng = 5.0, 49
    model = SCFTLatticeFluid([("star", ["A"=>4, "B"=>3]), ("solv", ["A"=>2])], [1.0, 1.2],
                             [0.0 3.0; 3.0 0.0]; rho0=1.0, kappa=15.0)
    structure = ClassicalDFT.Uniform1DCart((0.0, 0.0), [0.8, 0.2], [0.0, L], ng)
    system = ClassicalDFT.SCFTSystem(model, structure, ClassicalDFT.DFTOptions();
        mol_structure=Dict("star" => ClassicalDFT.custom_structure("AA(BBB)AA"),
                           "solv" => ClassicalDFT.custom_structure("AA")),
        ensemble=[:canonical, :grand_canonical], n_molecules=[0.8L/7, 0.0])
    ρ = ClassicalDFT.initialize_profiles(system)
    z = range(0, L, length=ng)
    _seed_cos!(view(ρ, :, 2), z, L, 0.05, 1)
    ClassicalDFT.converge!(system, ρ; _REG_OPTS..., maxit=12)
    return ρ
end

function _reg_wlc_rodcoil_ms()
    # Stiff/flexible WLC diblock with Maier-Saupe coupling on the stiff block, 1D.
    N, L, ng = 6, 12.0, 48
    model = SCFTWormLikeChainFluid([("rc", ["A"=>N, "B"=>N])], [1.0, 1.0], [8.0, 1.0],
                                   [0.0 2.0; 2.0 0.0]; rho0=1.0, kappa=20.0,
                                   nu=[1.0 0.0; 0.0 0.0], L_max=6)
    structure = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng)
    system = ClassicalDFT.SCFTSystem(model, structure, ClassicalDFT.DFTOptions();
        mol_structure=Dict("rc" => ClassicalDFT.custom_structure("A"^N * "B"^N)),
        ensemble=[:canonical], n_molecules=[L/(2N)])
    ρ = ClassicalDFT.initialize_profiles(system)
    z = range(0, L, length=ng)
    _seed_cos!(view(ρ, :, 1), z, L, 0.05,  1)
    _seed_cos!(view(ρ, :, 2), z, L, 0.05, -1)
    # Run to convergence: a fixed, unconverged iteration count picks up run-to-run
    # floating-point noise. (ν=1, χ=2 is chosen because it converges cleanly; at ν=2 the
    # pre-refactor solver stalls or hits NaN.)
    ClassicalDFT.converge!(system, ρ; _REG_OPTS..., tol=1e-8, maxit=3000)
    return ρ
end

function _reg_sequence_taper()
    # Tapered sequence strong enough to order into a lamella; converged (a fixed small
    # iteration count is round-off-sensitive for this case — Anderson amplifies it).
    Nc, L, ng = 20, 7.0, 65
    desc = taper_descriptor(0.0, 1.0, Nc; taper_fraction=0.4, taper_start=0.3)
    model = SCFTSequenceFluid(desc, 1.0; rho0=1.0, kappa=20.0,
                              gamma=(f1, f2) -> 900 * sum(abs2, f1 .- f2))
    structure = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng)
    system = SCFTSequenceSystem(model, structure; ensemble=:canonical, n_molecules=L/Nc)
    ρ = ClassicalDFT.initialize_profiles(system)
    z = range(0, L, length=ng)
    for k in 1:Nc
        _seed_cos!(view(ρ, :, k), z, L, 0.005, k <= Nc ÷ 2 ? 1 : -1)
    end
    ClassicalDFT.converge!(system, ρ; _REG_OPTS..., maxit=600)
    return ρ
end

const SCFT_REGRESSION_CASES = [
    "lamella_dgc"     => _reg_lamella_dgc,
    "star_blend_gc"   => _reg_star_blend_gc,
    "wlc_rodcoil_ms"  => _reg_wlc_rodcoil_ms,
    "sequence_taper"  => _reg_sequence_taper,
]

"""
Scalar fingerprint of a density array, per species: mean, then (sum of squared deviations
from the mean, and the deviation at three grid points). Deviations are reported separately
from the mean so that weakly perturbed states stay detectable at a relative tolerance.
"""
function scft_fingerprint(ρ)
    nd = ndims(ρ) - 1
    ns = size(ρ, nd + 1)
    out = Float64[]
    for α in 1:ns
        ρα = vec(collect(selectdim(ρ, nd + 1, α)))
        n = length(ρα)
        m = sum(ρα) / n
        δ = ρα .- m
        append!(out, (m, sum(abs2, δ), δ[1], δ[cld(n, 3)], δ[cld(2n, 3)]))
    end
    return out
end
