"""
    interfacial_tension(model::EoSModel,p,T,x,xx)

Calculate the interfacial tension of a liquid-liquid interface based on a given `model` and at conditions `p`, `T`, `x` and `xx`, where `x` and `xx` are the compositions of the two phases, respectively. This calculation assumes the interface is 1D and cartesian, and that the two bulk phases are in equilbrium. If the system is expected to form micelles, other methods should be used.

Example:
```julia
julia> model = PCSAFT(["water","hexane"])

julia> (n,_,_) = tp_flash(model,1e5,298.15,[0.5,0.5],RRTPFlash(equilibrium=:lle))

julia> interfacial_tension(model,1e5,298.15,n[1,:],n[2,:])
```
"""
function interfacial_tension(model::EoSModel,p,T,x,xx)
    L = length_scale(model)

    v1 = volume(model,p,T,x)
    v2 = volume(model,p,T,xx)

    ρ1 = x./v1
    ρ2 = xx./v2

    structure = TwoPhase1DCart((p, T), ρ1, ρ2,[-20L,20L], (201,))

    system = DFTSystem(model, structure)
    ρ = initialize_profiles(system)

    converge!(system,ρ)
    return interfacial_tension(system,ρ)/2
end

"""
    interfacial_tension(system::DFTSystem, ρ)

Calculate the interfacial tension of an based on a given `system` and density profile `ρ`. To get accurate results, the system should be converged first. 

Example:
```julia
julia> model = PCSAFT(["water","hexane"])

julia> (n,_,_) = tp_flash(model,1e5,298.15,[0.5,0.5],RRTPFlash(equilibrium=:lle))

julia> v1 = volume(model,1e5,298.15,n[1,:])

julia> v2 = volume(model,1e5,298.15,n[2,:])

julia> ρ1 = n[1,:]./v1

julia> ρ2 = n[2,:]./v2

julia> structure = TwoPhase1DCart((1e5, 298.15), ρ1, ρ2, [-10L, 10L], 201)

julia> system = DFTSystem(model, structure)

julia> ρ = initialize_profiles(system)  

julia> converge!(system, ρ)

julia> interfacial_tension(system, ρ)
```
"""
interfacial_tension(system::AbstractClassicalDFTSystem, ρ) = surface_tension(system, ρ)

export interfacial_tension