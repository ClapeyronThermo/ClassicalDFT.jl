"""
    FloryHuggins(chi)

Local Flory-Huggins contact interaction between species, with energy and mean field

```
U = (1/ρ₀) ∫ Σ_{α<β} χ_αβ ρ_α ρ_β dr,        w_α(r) += Σ_β (χ_αβ/ρ₀) ρ_β(r)
```

`chi` is a symmetric, zero-diagonal matrix in species order
(`model.groups.flattenedgroups`) or a `Dict` of species-name pairs, e.g.
`Dict(("A","B") => 1.5)`.
"""
struct FloryHuggins{M} <: SCFTInteractionModel
    chi::M
end

bind_interaction(int::FloryHuggins, species::Vector{String}) =
    FloryHuggins(_species_matrix(int.chi, species, "FloryHuggins chi"; zero_diagonal=true))

function add_field!(w, int::FloryHuggins, ρ, ρ_total, rho0, nd)
    FT = eltype(w)
    chi = int.chi
    n = size(chi, 1)
    for α in 1:n
        w_α = selectdim(w, nd + 1, α)
        for β in 1:n
            c = chi[α, β]
            iszero(c) && continue
            w_α .+= FT(c / rho0) .* selectdim(ρ, nd + 1, β)
        end
    end
    return w
end

function bulk_f(int::FloryHuggins, ρ::AbstractVector, rho0)
    chi = int.chi
    f = zero(eltype(ρ))
    for α in eachindex(ρ), β in eachindex(ρ)
        f += chi[α, β] * ρ[α] * ρ[β]
    end
    return f / (2 * rho0)
end

function field_energy(int::FloryHuggins, ρ, ρ_total, rho0, nd, per_∫, aux)
    chi = int.chi
    n = size(chi, 1)
    integrand = zero(ρ_total)
    for α in 1:n, β in (α + 1):n
        c = chi[α, β]
        iszero(c) && continue
        integrand .+= c .* selectdim(ρ, nd + 1, α) .* selectdim(ρ, nd + 1, β)
    end
    return per_∫(integrand) / rho0
end

export FloryHuggins
