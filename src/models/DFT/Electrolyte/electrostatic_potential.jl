abstract type ElectrostaticPotentialModel <: ExternalFieldModel end

struct ElectrostaticPotential{M} <: ElectrostaticPotentialModel
    ϵ_r::Float64
    map::M
end

export ElectrostaticPotential

"""
    ElectrostaticPotential(model::ElectrolyteModel, structure::DFTStructure, backend::Backend, ::Type{FP}=Float64)

External field representing the mean-field electrostatic (Coulomb) interaction between charged species in a Cartesian `DFTStructure`. Precomputes the Fourier-space Coulomb Green's-function kernel (scaled by the solvent's dielectric constant) that is convolved with the ionic charge density profile during `evaluate_external_field!`. This field is added automatically whenever an `ElectrolyteDFTSystem` is constructed, and should not typically need to be constructed directly by users.
"""
function ElectrostaticPotential(model::ElectrolyteModel, structure::DFTStructure, backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    (_, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ϵ_r = dielectric_constant(model.ionmodel.RSPmodel, 1., temperature, ρbulk)
    ngrid = structure.ngrid
    nd = length(ngrid)

    ω = structure_ω(structure, backend, FP)

    ω_norm = allocate(CPU(), FP, ngrid...)

    for kk in CartesianIndices(ngrid)
        ω_norm[kk] = norm(@view(ω[Tuple(kk)...,:]))
    end

    ω̄ = allocate(backend, FP, ngrid...)
    copyto!(ω̄,Adapt.adapt(typeof(ω̄), ω_norm))

    _c  = FP(N_A * e_c^2 / ϵ_0) / FP(ϵ_r)
    Ω   = @. (!iszero(ω̄)) / (FP(4)*π*π*ω̄^2 + iszero(ω̄)) * _c
    return ElectrostaticPotential(ϵ_r, Ω)
end

"""
    ElectrostaticPotential(model, structure::DFTStructure{N,Union{Cylindrical,Spherical}}, backend, FP)

Spherical/cylindrical (QDHT-based) counterpart of the Cartesian `ElectrostaticPotential`
constructor above. Reuses the same 3D-isotropic Coulomb Green's-function kernel formula
unchanged — a `z`-translation-invariant 3D charge distribution only excites the `k_z=0`
slice of the isotropic 3D Coulomb kernel, which has the same functional form, so no
rescaling is needed between the spherical and cylindrical cases. `ω̄=0` never occurs on
the QDHT grid, so the Cartesian version's zero-mode mask is unnecessary here.
"""
function ElectrostaticPotential(model::ElectrolyteModel, structure::Union{DFTStructByCoord{Cylindrical},DFTStructByCoord{Spherical}}, backend::Backend, ::Type{FP}=Float64) where {FP<:AbstractFloat}
    backend isa CPU || error("Spherical/cylindrical coordinate systems are CPU-only for now")
    (_, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ϵ_r = dielectric_constant(model.ionmodel.RSPmodel, 1., temperature, ρbulk)

    ω̄ = structure_ω(structure, backend, FP).ω̄

    _c = FP(N_A * e_c^2 / ϵ_0) / FP(ϵ_r)
    Ω  = @. _c / (FP(4)*π*π*ω̄^2)
    return ElectrostaticPotential(ϵ_r, Ω)
end


"""
    evaluate_external_field!(structure, external_field, model::ElectrolyteModel, ρ, δfδρ_res, P, iP, Vext)

Adds one extra step beyond the plain convolution: re-anchors `Vext` to the
domain's structural center grid point before folding it into `δfδρ_res`.

`convolve!`'s own kernel has `Ω(0)=0` (a standard periodic-Ewald device to
avoid a `k=0` divergence, not a physical reference choice), which leaves
`Vext` in a zero-*domain-average* gauge. `get_new_profile!`'s own
per-species target (`chem_pot_res_dens`) is built from the model's bulk
chemical potential evaluated only at `structure.ρbulk` -- which implicitly
assumes zero external field AT that state. For a genuinely two-phase
structure (e.g. `TwoPhase1DCart`, whose own convention places the dense
`ρbulk` phase at the domain's structural center, dilute at the edges), the
zero-domain-average gauge instead splits any real bulk-to-bulk potential
jump roughly symmetrically around zero, leaving a spurious nonzero field at
the `ρbulk` reference point itself. Anchoring at the domain's center index
is harmless for a spatially uniform (single-bulk) structure too: `Vext` is
already ~0 everywhere there (no charge gradient to drive it), so
subtracting its own center value is a no-op, not a shift -- confirmed
against this package's own uniform-bulk `ElectrolyteDFTSystem` tests
(Cartesian, spherical and cylindrical).
"""
function evaluate_external_field!(structure::DFTStructure,external_field::ElectrostaticPotentialModel,model::ElectrolyteModel,ρ,δfδρ_res,P,iP,Vext)
    temperature = structure.conditions[2]
    Z = model.charge
    ngrid = structure.ngrid
    bounds = structure.bounds
    L = [bounds[i][2]-bounds[i][1] for i in 1:length(bounds)]
    Vol = prod(L)
    nbeads = length(Z)
    nd = length(ngrid)
    # obtain charge profiles
    for i in 1:nbeads
        # println(i)
        if i == 1
            Vext .= selectdim(ρ,nd+1,i)*Z[i]
        else
            Vext .+= selectdim(ρ,nd+1,i)*Z[i]
        end
    end

    ϵ_r = external_field.ϵ_r
    map = external_field.map

    convolve!(Vext, Vext, map, P, iP, Vext)

    center_idx = CartesianIndex(ntuple(d -> (ngrid[d] + 1) ÷ 2, nd))
    # `Vext[center_idx]` (a single-element scalar `getindex`) is disallowed
    # on a GPU array -- confirmed directly under a CUDA `DFTOptions`
    # ("Scalar indexing is disallowed"). `center_idx:center_idx` builds a
    # 1-element `CartesianIndices` RANGE instead of a lone `CartesianIndex`,
    # so `Vext[center_idx:center_idx]` is an ordinary (GPU-safe) array-slicing
    # `getindex`; `Array(...)` then copies just that one element to host.
    center_val = Array(Vext[center_idx:center_idx])[1]
    Vext .-= center_val

    for i in 1:nbeads
        selectdim(δfδρ_res,nd+1,i) .+= Z[i]*Vext / k_B / temperature
    end
end

"""
    find_ψ_const(structure, external_field, model::ElectrolyteModel, ρ; maxit=200)

Newton solve for the constant potential shift `ψ0` enforcing the domain-
integrated electroneutrality constraint `Σ_k Z_k∫ρ_k dx = 0`. Capped at
`maxit` iterations: for a genuinely net-charged system, or an intermediate
(not-yet-converged) trial profile mid-SCF-iteration, this Newton solve can
plausibly land somewhere `dq` is pathologically small or the root doesn't
exist cleanly -- an uncapped loop would hang the entire outer `converge!`
call indefinitely with no diagnostic. Returns the best available `ψ0` (with
a warning) rather than looping forever if `maxit` is reached.
"""
function find_ψ_const(structure::DFTStructure,external_field::ElectrostaticPotentialModel,model::ElectrolyteModel,ρ;maxit::Int=200)
    Z = model.charge
    nbeads = length(Z)
    nd = length(structure.ngrid)
    ψ0 = 0.
    converged = false
    for _ in 1:maxit
        q = 0.
        dq = 0.
        for i in 1:nbeads
            q += sum(selectdim(ρ,nd+1,i)*Z[i])*exp(-Z[i]*ψ0)
            dq -= sum(selectdim(ρ,nd+1,i))*Z[i]^2*exp(-Z[i]*ψ0)
        end
        ψ0 -= q/dq
        if abs(q) < 1e-6
            converged = true
            break
        end
    end
    converged || @warn "find_ψ_const: did not converge within $maxit iterations; returning best available ψ0=$ψ0"
    return ψ0*k_B*structure.conditions[2]
end