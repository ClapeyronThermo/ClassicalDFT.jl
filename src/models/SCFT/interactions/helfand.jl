"""
    HelfandCompressibility(kappa)

Helfand's soft incompressibility penalty on the total segment density
`ρ₊ = Σ_α ρ_α`:

```
U = (κ/2) ∫ (ρ₊/ρ₀ − 1)² dr,        w_α(r) += (κ/ρ₀)(ρ₊(r)/ρ₀ − 1)
```

Large `kappa` approaches strict incompressibility.
"""
struct HelfandCompressibility <: SCFTInteractionModel
    kappa::Float64
end
HelfandCompressibility(kappa::Real) = HelfandCompressibility(Float64(kappa))

function add_field!(w, int::HelfandCompressibility, ρ, ctx)
    (; ρ_total, rho0, nd) = ctx
    FT = eltype(w)
    a = FT(int.kappa / rho0)
    inv_rho0 = FT(1 / rho0)
    for α in 1:size(w, nd + 1)
        w_α = selectdim(w, nd + 1, α)
        @. w_α += a * (ρ_total * inv_rho0 - one(FT))
    end
    return w
end

bulk_f(int::HelfandCompressibility, ρ::AbstractVector, rho0) =
    (int.kappa / (2 * rho0^2)) * (sum(ρ) - rho0)^2

function field_energy(int::HelfandCompressibility, ρ, ρ_total, ctx)
    (; per_∫, rho0) = ctx
    return per_∫((int.kappa / 2) .* (ρ_total ./ rho0 .- 1) .^ 2)
end

export HelfandCompressibility
