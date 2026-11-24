"""
    Incompressible(; lambda=1.0)

Strict incompressibility, `Σ_α ρ_α(r) = ρ₀` everywhere, enforced by a pressure field
`ξ(r)` added to every species' field: `w_α = F_α[ρ] + ξ`, where `F_α` is the sum of all the
other interactions' fields.

`ξ` is not a separate unknown: following Matsen's update for incompressible SCFT, it is
reconstructed each iteration from the current field `w` the densities came from,

```
ξ(r) = (1/n) Σ_γ [w_γ(r) − F_γ(r)] + λ (ρ₊(r)/ρ₀ − 1),
```

so the Anderson-accelerated unknown is still just `w`. At a fixed point every
`w_α − F_α` equals the same `ξ`, and the `λ` term then forces `ρ₊ = ρ₀`. `lambda` (`k_BT`
units) sets how hard each iteration pushes toward incompressibility; larger values reach
the constraint in fewer iterations but can destabilise the iteration.

Use instead of, not together with, [`HelfandCompressibility`](@ref). Contributes nothing
to the bulk free energy (the bulk densities must already satisfy `Σρ = ρ₀`) and nothing
extra to the free energy at convergence, where `∫ξ(ρ₀ − ρ₊) = 0`.
"""
struct Incompressible <: SCFTInteractionModel
    lambda::Float64
end
Incompressible(; lambda::Real = 1.0) = Incompressible(Float64(lambda))

is_constraint(::Incompressible) = true

function validate_interaction(::Incompressible, model)
    any(i -> i isa HelfandCompressibility, model.interactions) && throw(ArgumentError(
        "Incompressible and HelfandCompressibility both constrain the total density; use one or the other."))
    count(i -> i isa Incompressible, model.interactions) == 1 || throw(ArgumentError(
        "At most one Incompressible interaction is allowed."))
    return nothing
end

preallocate_interaction(::Incompressible, system, ρ) =
    similar(selectdim(ρ, dimension(system) + 1, 1))

function add_field!(w, int::Incompressible, ρ, ctx)
    (; ρ_total, rho0, nd, w_current) = ctx
    ξ = ctx.cache === nothing ? similar(ρ_total) : ctx.cache
    FT = eltype(w)
    n = size(w, nd + 1)
    λ = FT(int.lambda)
    inv_rho0 = FT(1 / rho0)
    # On entry `w` holds Σ of every other interaction's field, F_α.
    @. ξ = λ * (ρ_total * inv_rho0 - one(FT))
    if w_current !== nothing
        inv_n = FT(1 / n)
        for γ in 1:n
            ξ .+= inv_n .* (selectdim(w_current, nd + 1, γ) .- selectdim(w, nd + 1, γ))
        end
    end
    for α in 1:n
        selectdim(w, nd + 1, α) .+= ξ
    end
    return w
end

export Incompressible
