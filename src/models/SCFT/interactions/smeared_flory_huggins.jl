"""
    SmearedFloryHugginsChi(chi, a)

Finite-range Flory-Huggins interaction: χ acts between Gaussian-smeared densities
`ρ̃ = h * ρ`, with `h(r) ∝ exp(-|r|²/(2a²))` normalized to unit integral:

```
U = (1/ρ₀) ∫ Σ_{α<β} χ_αβ ρ̃_α ρ̃_β dr,     w_α(r) += Σ_β (χ_αβ/ρ₀) (h * h * ρ_β)(r)
```

In Fourier space `ĥ(k) = exp(-k²a²/2)`, so `ĥ(0) = 1`: a uniform system sees exactly the
local [`FloryHugginsChi`](@ref) interaction (identical bulk free energy and bulk fields),
and `a → 0` recovers the local model everywhere. `a > 0` softens interfaces and removes
the grid-scale ultraviolet sensitivity of a contact interaction. `chi` takes the same
forms as for `FloryHugginsChi`; `a` has units of length.

Requires a periodic Cartesian structure (it convolves with FFTs).
"""
struct SmearedFloryHugginsChi{M} <: SCFTInteractionModel
    chi::M
    a::Float64
end
SmearedFloryHugginsChi(chi, a::Real) = SmearedFloryHugginsChi(chi, Float64(a))

bind_interaction(int::SmearedFloryHugginsChi, species::Vector{String}) =
    SmearedFloryHugginsChi(_species_matrix(int.chi, species, "SmearedFloryHugginsChi chi"; zero_diagonal=true), int.a)

# Squared angular wavenumber |k|² = (2π)²|ν|² on the R2C (half-complex) grid of `structure`.
function _rfft_k2(structure, ::Type{FP}) where FP
    nd = dimension(structure)
    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)
    lb1, ub1 = bounds(structure, 1)
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)
    k2 = zeros(FP, rfft_ngrid...)
    for d in 1:nd
        ν = d == 1 ? rfftfreq(ngrid[1], ngrid[1] / (ub1 - lb1)) : ω̂[d]
        k2 .+= reshape(FP.((2π .* ν) .^ 2), ntuple(i -> i == d ? rfft_ngrid[d] : 1, nd))
    end
    return k2
end

function preallocate_interaction(int::SmearedFloryHugginsChi, system, ρ)
    nd = dimension(system)
    FP = eltype(ρ)
    CT = Complex{FP}
    ngrid = system.structure.ngrid
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)
    k2 = _rfft_k2(system.structure, FP)
    device = system.options.device
    kernel = Adapt.adapt(device, CT.(exp.(-k2 .* FP(int.a)^2)))   # ĥ², for the field
    buf_r = similar(ρ, FP, ngrid...)
    buf_c = similar(ρ, CT, rfft_ngrid...)
    smeared = similar(ρ)                                           # (h*h*ρ_β) per species
    P  = plan_rfft(buf_r, 1:nd)
    iP = plan_irfft(buf_c, ngrid[1], 1:nd)
    return (; kernel, buf_r, buf_c, smeared, P, iP)
end

function add_field!(w, int::SmearedFloryHugginsChi, ρ, ctx)
    (; rho0, nd) = ctx
    c = ctx.cache
    FT = eltype(w)
    chi = int.chi
    n = size(chi, 1)
    for β in 1:n
        any(!iszero, view(chi, :, β)) || continue
        convolve!(selectdim(c.smeared, nd + 1, β), selectdim(ρ, nd + 1, β),
                  c.kernel, c.P, c.iP, c.buf_r, c.buf_c)
    end
    for α in 1:n
        w_α = selectdim(w, nd + 1, α)
        for β in 1:n
            iszero(chi[α, β]) && continue
            w_α .+= FT(chi[α, β] / rho0) .* selectdim(c.smeared, nd + 1, β)
        end
    end
    return w
end

bulk_f(int::SmearedFloryHugginsChi, ρ::AbstractVector, rho0) = bulk_f(FloryHugginsChi(int.chi), ρ, rho0)

function field_energy(int::SmearedFloryHugginsChi, ρ, ρ_total, ctx)
    (; rho0, nd, per_∫, structure) = ctx
    chi = int.chi
    n = size(chi, 1)
    ngrid = size(ρ_total)
    ĥ = exp.(-_rfft_k2(structure, Float64) .* (int.a^2 / 2))
    ρ̃ = [irfft(ĥ .* rfft(Array{Float64}(selectdim(ρ, nd + 1, α)), 1:nd), ngrid[1], 1:nd) for α in 1:n]
    integrand = zeros(ngrid...)
    for α in 1:n, β in (α + 1):n
        iszero(chi[α, β]) && continue
        integrand .+= chi[α, β] .* ρ̃[α] .* ρ̃[β]
    end
    return per_∫(integrand) / rho0
end

export SmearedFloryHugginsChi
