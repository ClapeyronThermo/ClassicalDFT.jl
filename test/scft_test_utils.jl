# Shared helpers for the SCFT test files. Each file includes this once:
#     @isdefined(SCFT_TEST_UTILS) || include("scft_test_utils.jl")
using Test, ClassicalDFT, Clapeyron, Random
using KernelAbstractions: CPU

const SCFT_TEST_UTILS = true
const CDFT = ClassicalDFT

"""
    scft_system(seq; chain, interactions, L=6.0, ng=24, L_max=6, rho0=1.0, n_molecules, propagator)

A 1D periodic, canonical SCFT system with one molecule type whose topology and group
counts come from the `custom_structure` string `seq` (e.g. `"AAABBB"`, `"AA(BB)(BBB)A"`).
`n_molecules` defaults to filling the box to `rho0`. `propagator`, when given, is a
function `system -> propagator` that replaces the automatically chosen one.
"""
function scft_system(seq; chain=GaussianChain(), interactions=(HelfandCompressibility(20.0),),
                     L=6.0, ng=24, L_max=6, rho0=1.0,
                     n_molecules=rho0 * L / count(isletter, seq), propagator=nothing)
    letters = unique(filter(isletter, seq))
    gl = [("mol", [string(l) => count(==(l), seq) for l in letters])]
    m = SCFTModel(gl; rho0, chain, interactions, L_max)
    st = CDFT.Uniform1DCart((0.0, 0.0), [rho0], [0.0, L], ng)
    sys = CDFT.SCFTSystem(m, st, CDFT.DFTOptions(); mol_structure=Dict("mol" => CDFT.custom_structure(seq)),
                          ensemble=[:canonical], n_molecules=[n_molecules])
    propagator === nothing && return sys
    return CDFT.SCFTSystem(sys.model, sys.species, sys.structure, propagator(sys), sys.options, sys.external_field)
end

box_length(sys) = sys.structure.ngrid[1] * CDFT.structure_dz(sys.structure)[1]

"""Initial profile with species 1 raised and species 2 lowered by `amp·cos(2πz/L)`."""
function lamella_seed(sys; amp=0.05)
    ρ = CDFT.initialize_profiles(sys)
    size(ρ, 2) >= 2 || return ρ
    L = box_length(sys)
    z = range(0, L, length=size(ρ, 1))
    ρ[:, 1] .+= amp .* cos.(2π .* z ./ L)
    ρ[:, 2] .-= amp .* cos.(2π .* z ./ L)
    return ρ
end

"""Converge `sys` from [`lamella_seed`](@ref); returns `(ρ, result)`."""
function solve_lamella(sys; amp=0.05, tol=1e-10, maxit=3000, kw...)
    ρ = lamella_seed(sys; amp)
    return ρ, CDFT.converge!(sys, ρ; verbose=false, tol, maxit, kw...)
end

contrast(ρ, α=1) = maximum(ρ[:, α]) - minimum(ρ[:, α])
max_compressibility_error(ρ, rho0=1.0) = maximum(abs.(sum(ρ, dims=2) .- rho0))

"""
    propagate_fields(sys, w) -> (; Q, ρ, q_in, q_out, w_bulk)

Run the propagators once on the field `w` (no self-consistency), then the partition
functions and densities.
"""
function propagate_fields(sys, w)
    w_bulk = CDFT.compute_bulk_fields(sys.model, CDFT.compute_bulk_densities(sys))
    ρ = zero(w)
    cache = CDFT.preallocate_propagator(sys, sys.propagator, ρ, CPU())
    CDFT.propagate!(sys, ρ, w, cache; w_bulk=w_bulk)
    q_in, q_out = CDFT.cache_q_in(cache), CDFT.cache_q_out(cache)
    Q = CDFT.compute_partition_functions(sys, w, w_bulk, q_in, CDFT.structure_dz(sys.structure))
    CDFT.compute_densities!(sys, w, w_bulk, q_in, q_out, Q, ρ)
    return (; Q, ρ, q_in, q_out, w_bulk)
end

"""Uniaxial order along z per species, `S = (3⟨u_z²⟩ − 1)/2`, from `R = ρ⟨uu⟩` (1D grid)."""
uniaxial_order(R, ρ) = [(3 * sum(R[:, α, 3]) / sum(ρ[:, α]) - 1) / 2 for α in axes(ρ, 2)]

"""
Independent reference for rigid rods: classical Maier-Saupe with total coupling `a` per
rod (weight `exp(a S P₂(cosθ))`), and a compressible rod melt (N beads, Helfand κ,
Maier-Saupe ν) built on it, all by Simpson quadrature over cosθ.
"""
module RodMeltReference
    const xs = range(-1, 1, length=20001)
    const sw = (h = step(xs); w = fill(2h / 3, length(xs)); w[2:2:end-1] .= 4h / 3; w[1] = w[end] = h / 3; w ./ 2)
    P2(x) = (3x^2 - 1) / 2
    logZ(a) = (e = a .* P2.(xs); m = maximum(e); m + log(sum(sw .* exp.(e .- m))))
    function S_self(a; S=0.8)
        for _ in 1:100000
            e = a * S .* P2.(xs); w = sw .* exp.(e .- maximum(e))
            Sn = sum(P2.(xs) .* w) / sum(w)
            abs(Sn - S) < 1e-14 && return Sn
            S = Sn
        end
        return S
    end
    f(ρm, S; N, κ, ν, ρ0) = (ρs = N * ρm; ρm * (log(ρm) - 1) + κ / (2ρ0^2) * (ρs - ρ0)^2 - ν * ρs^2 / (4ρ0) +
                             ν * ρs^2 * S^2 / (2ρ0) - ρm * logZ(N * ν * ρs * S / ρ0))
    μ(ρm, S; p...) = (h = 1e-7; (f(ρm + h, S; p...) - f(ρm - h, S; p...)) / 2h)
    P(ρm, S; p...) = ρm * μ(ρm, S; p...) - f(ρm, S; p...)
    function coexistence(ρI, ρN; N, κ, ν, ρ0)
        p = (; N, κ, ν, ρ0)
        r(y) = (SN = S_self(N * ν * N * exp(y[2]) / ρ0);
                [μ(exp(y[1]), 0.0; p...) - μ(exp(y[2]), SN; p...), P(exp(y[1]), 0.0; p...) - P(exp(y[2]), SN; p...)])
        y = [log(ρI), log(ρN)]
        for _ in 1:60
            r0 = r(y); maximum(abs, r0) < 1e-11 && break
            J = hcat([(r(y .+ h) .- r(y .- h)) ./ 2e-6 for h in ([1e-6, 0], [0, 1e-6])]...)
            y .-= J \ r0
        end
        return exp.(y), S_self(N * ν * N * exp(y[2]) / ρ0)
    end
end
