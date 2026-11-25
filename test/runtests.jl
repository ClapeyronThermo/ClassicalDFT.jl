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
    include("test_scft_regression.jl")
    include("test_scft_model.jl")
    include("test_scft_composite.jl")
    include("test_scft_orientation.jl")
    include("test_scft_sequence.jl")
    include("test_wlc.jl")
    # GPU backend parity (test_wlc_gpu.jl) is not run for now; include it again to test
    # Metal/CUDA. It self-skips without a functional backend.
    include("test_methods.jl")
    include("test_coordinate_systems.jl")
    include("test_two_phase.jl")
    include("test_morphology.jl")
    include("test_anderson.jl")
end