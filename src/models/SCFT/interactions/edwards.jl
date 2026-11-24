"""
    EdwardsExcludedVolume(v)

Edwards excluded-volume (δ-function pseudopotential) interaction, the standard
good-solvent / semidilute-solution term:

```
U = ½ ∫ Σ_αβ v_αβ ρ_α ρ_β dr,        w_α(r) += Σ_β v_αβ ρ_β(r)
```

`v` (units of volume, `k_BT = 1`) is a single `Real` applied to every species pair
(including like pairs), a symmetric matrix in species order, or a `Dict` of species-name
pairs. Unlike [`FloryHugginsChi`](@ref), like-pair entries `v[α,α]` are meaningful. With no
compressibility term in the model, `EdwardsExcludedVolume` alone sets the osmotic
equation of state, `Π = ρ_chains + ½ Σ v_αβ ρ_α ρ_β`.
"""
struct EdwardsExcludedVolume{M} <: SCFTInteractionModel
    v::M
end

bind_interaction(int::EdwardsExcludedVolume{<:Real}, species::Vector{String}) =
    EdwardsExcludedVolume(fill(Float64(int.v), length(species), length(species)))
bind_interaction(int::EdwardsExcludedVolume, species::Vector{String}) =
    EdwardsExcludedVolume(_species_matrix(int.v, species, "EdwardsExcludedVolume v"))

function add_field!(w, int::EdwardsExcludedVolume, ρ, ctx)
    nd = ctx.nd
    FT = eltype(w)
    v = int.v
    n = size(v, 1)
    for α in 1:n
        w_α = selectdim(w, nd + 1, α)
        for β in 1:n
            iszero(v[α, β]) && continue
            w_α .+= FT(v[α, β]) .* selectdim(ρ, nd + 1, β)
        end
    end
    return w
end

function bulk_f(int::EdwardsExcludedVolume, ρ::AbstractVector, rho0)
    v = int.v
    f = zero(eltype(ρ))
    for α in eachindex(ρ), β in eachindex(ρ)
        f += v[α, β] * ρ[α] * ρ[β]
    end
    return f / 2
end

function field_energy(int::EdwardsExcludedVolume, ρ, ρ_total, ctx)
    (; rho0, nd, per_∫) = ctx
    v = int.v
    n = size(v, 1)
    integrand = zero(ρ_total)
    for α in 1:n, β in 1:n
        iszero(v[α, β]) && continue
        integrand .+= v[α, β] .* selectdim(ρ, nd + 1, α) .* selectdim(ρ, nd + 1, β)
    end
    return per_∫(integrand) / 2
end

export EdwardsExcludedVolume
