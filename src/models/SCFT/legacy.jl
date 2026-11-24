#=
Legacy constructors. Each builds an SCFTModel from the pre-modular positional API, so
existing scripts, tests and docs keep working. New code should call SCFTModel directly.
=#

"""
    SCFTLatticeFluid(grouplist, b, chi; rho0, kappa, idealmodel=BasicIdeal, references=String[])

Gaussian-chain SCFT model with Flory-Huggins `chi` and Helfand compressibility `kappa`.
Equivalent to

```julia
SCFTModel(grouplist; rho0, chain = [GaussianChain(b=bα) for bα in b],
          interactions = (HelfandCompressibility(kappa), FloryHugginsChi(chi)))
```

`b` and `chi` are indexed by species (`groups.flattenedgroups`). Returns an
[`SCFTModel`](@ref).
"""
function SCFTLatticeFluid(grouplist, b::AbstractVector, chi::AbstractMatrix;
                          rho0::Real, kappa::Real, idealmodel = BasicIdeal, references = String[])
    return SCFTModel(grouplist; rho0, idealmodel, references,
        chain = SCFTChainModel[GaussianChain(b = bα) for bα in b],
        interactions = (HelfandCompressibility(kappa), FloryHugginsChi(chi)))
end

"""
    SCFTWormLikeChainFluid(grouplist, b, lp, chi; rho0, kappa, nu=nothing, L_max=8,
                           idealmodel=BasicIdeal, references=String[])

Worm-like-chain SCFT model: every species is a [`WormLikeChain`](@ref) with bond length
`b[α]` and persistence length `lp[α]` (`Inf` = rigid), with Flory-Huggins `chi`, Helfand
`kappa`, and (when `nu` is given) a [`MaierSaupe`](@ref) orientational coupling. Equivalent to

```julia
SCFTModel(grouplist; rho0, L_max, chain = [WormLikeChain(b=bα, lp=lpα) for (bα, lpα) in zip(b, lp)],
          interactions = (HelfandCompressibility(kappa), FloryHugginsChi(chi), MaierSaupe(nu)))
```

Returns an [`SCFTModel`](@ref).
"""
function SCFTWormLikeChainFluid(grouplist, b::AbstractVector, lp::AbstractVector, chi::AbstractMatrix;
                                rho0::Real, kappa::Real, nu::Union{AbstractMatrix,Nothing} = nothing,
                                L_max::Int = 8, idealmodel = BasicIdeal, references = String[])
    ints = (HelfandCompressibility(kappa), FloryHugginsChi(chi))
    nu === nothing || (ints = (ints..., MaierSaupe(nu)))
    return SCFTModel(grouplist; rho0, L_max, idealmodel, references,
        chain = SCFTChainModel[WormLikeChain(b = bα, lp = lpα) for (bα, lpα) in zip(b, lp)],
        interactions = ints)
end

export SCFTLatticeFluid, SCFTWormLikeChainFluid
