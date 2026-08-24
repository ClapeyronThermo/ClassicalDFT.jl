using Clapeyron: BasicIdealModel

"""
    F_ideal(system::DFTSystem, ρ)

Obtain the ideal free energy of the system for a given profile `ρ`.

The output is a scalar of units J.
"""
function F_ideal(system::AbstractClassicalDFTSystem,ρ)
    model = system.model
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    _bounds = system.structure.bounds

    n = zeros(ngrid...,length(model))
    for i in @comps
        for k in @chain(i)
            selectdim(n,nd+1,i) .+= Adapt.adapt(CPU(), selectdim(ρ,nd+1,k))/system.species.nbeads[i]
        end
    end
    
    f(x) = f_ideal(system,model.idealmodel,x)    

    ϕ = zeros(ngrid...)
    
    for j in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
        ϕ[j...] = f(@view n[j...,:])
    end

    return ∫(ϕ,system.structure)
end

function f_ideal(system::AbstractClassicalDFTSystem,model::BasicIdealModel,n)
    T = system.structure.conditions[2]
    ∑f = zero(T + first(n))
    lnT = log(T)
    return @sum(N_A*n[i]*(log(n[i]) - 1.5*lnT-1))
end