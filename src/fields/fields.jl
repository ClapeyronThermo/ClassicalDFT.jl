abstract type ScalarField <: DFTField end
abstract type VectorField <: DFTField end

include("weighted_densities.jl")

_field_len(field::ScalarField,nd::Int) = 1
_field_len(field::VectorField,nd::Int) = nd

function field_indices(fields::F,nd) where F
    ix = map(Base.Fix2(_field_len,nd),fields)
    cx = cumsum(ix)
    istart = cx .-  ix .+ 1
    iend = cx
    return range.(istart,iend)
end

field_indices(system::AbstractClassicalDFTSystem) = field_indices(system.fields,dimension(system))

"""
    evaluate_field!(system::AbstractClassicalDFTSystem, ρ, n, in_buf, out_buf, P, iP)

This function will obtain every field used in the system (listed in `system.fields`), writing the results in-place into `n`. The output has the dimensions `(ngrid...,nb)`, where `ngrid` is the number of grid points and `nb` is the number of beads in the model. `in_buf`/`out_buf` are scratch buffers and `P`/`iP` are the (forward/inverse) transform plans used for the underlying convolutions.

This is the dispatcher function that will call `evaluate_field!(system,field,ρ,...)` for each field in the system.
"""
function evaluate_field!(system::AbstractClassicalDFTSystem, ρ, n, in_buf, out_buf, P, iP)
    fields = system.fields
    ngrid = system.structure.ngrid
    nd = length(ngrid)

    foreach(field_indices(system),fields) do k,field
        # println("Evaluating field: ",field.type)
        if field isa ScalarField
            evaluate_field!(system,field,ρ,selectdim(n,nd+1,k[1]), in_buf, out_buf, P, iP)
        else
            evaluate_field!(system,field,ρ,selectdim(n,nd+1,k), in_buf, out_buf, P, iP)
        end
    end  
    
    return n
end

"""
    integrate_field!(system::AbstractClassicalDFTSystem, δf, δfδρ_res, in_buf, P, iP)

This function will obtain, for all fields, the functional derivative for each species / bead, accumulating the result in-place into `δfδρ_res`. `δfδρ_res` has the dimensions `(ngrid...,nb)`, where `ngrid` is the number of grid points and `nb` is the number of beads in the model. `in_buf` is a scratch buffer and `P`/`iP` are the (forward/inverse) transform plans used for the underlying convolutions.

This is the dispatcher function that will call `integrate_field!(system,field,...)` for each field in the system.
"""
function integrate_field!(system::AbstractClassicalDFTSystem, δf, δfδρ_res, in_buf, P, iP)
    fields = system.fields
    ngrid = system.structure.ngrid
    nd = length(ngrid)

    foreach(field_indices(system),fields) do k,field
        if field isa ScalarField
            integrate_field!(system,field,selectdim(δf,nd+1,k[1]), δfδρ_res, in_buf, P, iP)
        else
            integrate_field!(system,field,selectdim(δf,nd+1,k), δfδρ_res, in_buf, P, iP)
        end
    end
end