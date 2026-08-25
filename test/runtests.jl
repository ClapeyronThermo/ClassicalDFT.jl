using Test
t1 = @elapsed using ClassicalDFT
using Clapeyron
using GCIdentifier
using ChemicalIdentifiers
@info "Loading ClassicalDFT took $(round(t1,digits = 2)) seconds"

@testset verbose = true "ClassicalDFT" begin
    include("test_models.jl")
    include("test_dgt.jl")
    include("test_scft.jl")
    include("test_methods.jl")
    include("test_coordinate_systems.jl")
    include("test_two_phase.jl")
    include("test_morphology.jl")
    include("test_anderson.jl")
end