"""
    ClassicalDFT

A comprehensive library of classical Density Functional Theory (ClassicalDFT) models, along with a simple framework for developing your own. `ClassicalDFT` builds on top of [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) models (`PCSAFT`, `SAFTVRMie`, `SAFTgammaMie`, electrolyte models, ...) to compute inhomogeneous density profiles, surface/interfacial tensions and adsorption isotherms.

Example usage:
```julia
julia> using Clapeyron, ClassicalDFT

julia> model = PCSAFT(["water","octane"])

julia> interfacial_tension(model,1e5,298.15,[0.5,0.5])
```
"""
module ClassicalDFT

using LinearAlgebra
using NLSolvers, FFTW, LinearAlgebra
import AbstractFFTs: Plan
using Clapeyron
using Clapeyron: d, N_A, k_B, R̄, e_c, ϵ_0
using Clapeyron: @comps, @groups
using Clapeyron: Solvers, log, sqrt, log1p, PackedVofV, sparse, SparseMatrixCSC
using Clapeyron: assoc_similar, assoc_matrix_solve, assoc_options, assoc_pair_length
using Clapeyron: issite, compute_index, complement_index, indices, diagvalues
import Clapeyron.Solvers
import Clapeyron: ElectrolyteModel
using Clapeyron: SingleComp, PeTSModel, epsilon_LorentzBerthelot!
using StaticArrays
# using SIAMFANLEquations
using KernelAbstractions, Adapt, Enzyme
using Hankel

#if !isdefined(Clapeyron,Symbol("@sum"))
    include("utils/sum.jl")
#else
    #using Clapeyron: @sum
#end

include("base/base.jl")

include("utils/base.jl")
include("utils/connectivity.jl")
include("utils/expand_model.jl")
include("utils/integrals.jl")
include("utils/anderson.jl")
include("utils/matmul.jl")
include("utils/plot_labels.jl")
include("utils/plot_style.jl")

include("fields/fields.jl")
include("models/models.jl")
include("structure/structure.jl")
#TODO, port this to Clapeyron

include("propagator/propagator.jl")

include("methods/converge.jl")
include("methods/surface_tension.jl")
include("methods/interfacial_tension.jl")
include("methods/adsorption.jl")

export MolStructure, SMILESStructure, CustomStructure, smiles, custom_structure

end