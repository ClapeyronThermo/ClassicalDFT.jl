# Block-copolymer microphase morphology seed profiles (BCC, Hex, Gyroid, Lamellar).
#
# Unlike two_phase.jl (one scalar profile per *component*, transitioning between two bulk
# phases), these seed a single periodic unit cell in which *different groups within the
# same component* (e.g. "A"/"B" from a `custom_structure`) enrich in different spatial
# domains, following standard leading-order Fourier/level-set SCFT initial guesses
# (Matsen & Bates; Fredrickson, "Equilibrium Theory of Inhomogeneous Polymers").

_group_letter(name::String) = split(name, "_")[1]

# +1 for beads in a core_groups group, -1 (matrix) otherwise. One entry per flattened bead
# index (model.groups.flattenedgroups), in the same order @chain(i) iterates.
function _domain_sign(model::EoSModel, core_groups::Vector{String})
    hasfield(typeof(model), :groups) || error(
        "Block-copolymer morphology structures require a group-contribution model with " *
        "named groups (e.g. HeterogcPCPSAFT built with a `custom_structure`/`smiles` " *
        "mol_structure) — got $(typeof(model)), which has no `groups` field.")
    letters = _group_letter.(model.groups.flattenedgroups)
    unknown = setdiff(core_groups, unique(letters))
    isempty(unknown) || error("core_groups $(unknown) not found among this model's groups $(unique(letters)).")
    return [letter in core_groups ? 1.0 : -1.0 for letter in letters]
end

# Coordinate arrays shifted to start at 0 (one per dimension, each already broadcast to the
# full ngrid shape via get_coords), plus the box length along each dimension.
function _morph_coords(structure::DFTStructByType{<:BlockCopolymerMorphology})
    nd = dimension(structure)
    Z = get_coords(structure)
    lb_ub = ntuple(d -> bounds(structure,d), nd)
    coords = ntuple(d -> selectdim(Z, nd+1, d) .- lb_ub[d][1], nd)
    Ls = ntuple(d -> lb_ub[d][2]-lb_ub[d][1], nd)
    return coords, Ls
end

function lamellar_ψ(structure)
    coords, Ls = _morph_coords(structure)
    X, Lx = coords[1], Ls[1]
    n = structure.topology.periods
    f = structure.topology.core_fraction
    # A core-domain plateau (ψ=+1) of width f*period, centered on X=0 (mod period, matching
    # the old cos(2π*n*X/Lx)'s peak location), and a matrix plateau (ψ=-1) of width
    # (1-f)*period filling the rest -- at f=0.5 this is the same 50/50 split the plain
    # cosine gave, just as a step rather than a smooth ramp. A single global sinusoid can't
    # do this: reparametrizing its phase only moves where it crosses zero relative to the
    # segment BOUNDARIES, not the width of the positive region, since any point-symmetric
    # +1->-1 ramp crosses zero at its own segment's midpoint regardless of that segment's
    # width. This is only a seed for the SCFT iteration to relax from, so the discontinuity
    # here is harmless.
    q = @. mod(n*X/Lx + f/2, 1.0)
    return @. ifelse(q < f, 1.0, -1.0)
end

function hex_ψ(structure)
    coords, Ls = _morph_coords(structure)
    X, Y = coords[1], coords[2]
    Lx, Ly = Ls[1], Ls[2]
    n = structure.topology.periods
    return @. (1/3)*(cos(2π*n*(X/Lx + Y/Ly)) + cos(2π*n*(X/Lx - Y/Ly)) + cos(4π*n*Y/Ly))
end

function bcc_ψ(structure)
    coords, Ls = _morph_coords(structure)
    X, Y, Z = coords
    L = Ls[1]
    n = structure.topology.periods
    return @. (1/6)*(cos(2π*n*(X+Y)/L) + cos(2π*n*(X-Y)/L) +
                      cos(2π*n*(Y+Z)/L) + cos(2π*n*(Y-Z)/L) +
                      cos(2π*n*(Z+X)/L) + cos(2π*n*(Z-X)/L))
end

function gyroid_ψ(structure)
    coords, Ls = _morph_coords(structure)
    X, Y, Z = coords
    L = Ls[1]
    n = structure.topology.periods
    return @. (sin(2π*n*X/L)*cos(2π*n*Y/L) + sin(2π*n*Y/L)*cos(2π*n*Z/L) + sin(2π*n*Z/L)*cos(2π*n*X/L)) / 1.5
end

# Shared fill: ρ_k = ρbulk[component] * (1 + amplitude * sign_k * ψ). amplitude < 1
# guarantees strict positivity everywhere, regardless of ψ's local value.
function _fill_morphology!(ρ, structure::DFTStructByType{<:BlockCopolymerMorphology}, model, device, ::Type{FP}, ψ) where FP<:AbstractFloat
    sign = _domain_sign(model, structure.topology.core_groups)
    ρbulk = structure.ρbulk
    A = structure.topology.amplitude
    nd = dimension(structure)
    for i in @comps
        for j in @chain(i)
            ρ_points = @. ρbulk[i]*(1 + A*sign[j]*ψ)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end

morphology_ψ(structure::DFTStructByType{BlockCopolymerMorphology{:Lamellar}}) = lamellar_ψ(structure)
morphology_ψ(structure::DFTStructByType{BlockCopolymerMorphology{:HexLattice}}) = hex_ψ(structure)
morphology_ψ(structure::DFTStructByType{BlockCopolymerMorphology{:BodyCenteredCubic}}) = bcc_ψ(structure)
morphology_ψ(structure::DFTStructByType{BlockCopolymerMorphology{:Gyroid}}) = gyroid_ψ(structure)

function initialize_profiles(model::EoSModel, structure::DFTStructure{N,Cartesian,<:BlockCopolymerMorphology}, species, device, ::Type{FP}=Float64) where {N,FP<:AbstractFloat}
    ρ = allocate(device, FP, structure.ngrid..., sum(species.nbeads))
    return _fill_morphology!(ρ, structure, model, device, FP, morphology_ψ(structure))
end