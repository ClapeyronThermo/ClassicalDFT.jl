"""
    adsorption(system::AbstractClassicalDFTSystem, ρ)

Calculate the (per-species) adsorption of a converged density profile `ρ` at a surface, defined as the excess amount of each species per unit volume of the accessible domain. To get accurate results, the system should be converged first.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> surface = Steele(["graphite"])

julia> adsorption(model, surface, 1e5, 298.15)
```
"""

# Volume of the domain `∫` integrates over, for each coordinate system: for Cartesian this is
# literally the product of the bound lengths (matching `∫`'s plain `dz`/`dx dy`/... measure),
# but `bounds` on a radial (Cylindrical/Spherical) structure holds the *radial* extent
# `(r_min, r_max)`, not a literal length -- the accessible volume there is the disk area (per
# unit axial length, matching `∫`'s `2π∫r dr` convention) or the sphere volume (matching `∫`'s
# `4π∫r² dr`), not `r_max - r_min` itself.
_domain_volume(structure::DFTStructByCoord{Cartesian}) = prod(b[2]-b[1] for b in structure.bounds)
_domain_volume(structure::DFTStructByCoord{Cylindrical}) = π*(structure.bounds[1][2]^2 - structure.bounds[1][1]^2)
_domain_volume(structure::DFTStructByCoord{Spherical}) = (4π/3)*(structure.bounds[1][2]^3 - structure.bounds[1][1]^3)

function adsorption(system,ρ)
    # Integrate over all profiles
    nc = length(system.model)
    nd = dimension(system)
    V = _domain_volume(system.structure)
    return [∫(selectdim(ρ,nd+1,i)[:],system.structure)/V for i in 1:nc]
end

"""
    adsorption(model::EoSModel, surface::ExternalFieldModel, p, T, n=[1.0])

Calculate the adsorption of a given `model` at conditions `p`, `T` and bulk composition `n`, next to an external field `surface` (e.g. a `Steele` wall). This is a blind calculation that assumes the domain is 1D and cartesian, sets up the system between the wall and the edge of `surface`'s `width`, converges the density profile, and then integrates it via the `AbstractClassicalDFTSystem` method above.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> surface = Steele(["graphite"])

julia> adsorption(model, surface, 1e5, 298.15)
```
"""
function adsorption(model::EoSModel, surface::ExternalFieldModel, p, T, n=[1.0])
    L = ClassicalDFT.length_scale(model)

    width = surface.params.width[1]
    bounds = [0.7L,width-0.7L]

    v = volume(model,p,T,n)
    ρ = n./v

    structure = ClassicalDFT.Uniform1DCart((p, T), ρ, bounds, (201,))

    system = ClassicalDFT.DFTSystem(model, structure, surface)

    ρ = initialize_profiles(system)

    converge!(system,ρ)

    return adsorption(system, ρ)
end

export adsorption