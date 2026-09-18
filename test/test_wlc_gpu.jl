using Test, ClassicalDFT

# Backend-parity tests for the WLC propagator's KernelAbstractions support -- checks that
# a converged density profile AND orientation_order_parameter's output agree between a
# GPU backend and CPU, run on the SAME model/structure. Self-skips (registering no testset
# at all, not a spurious skipped/broken entry) on any machine without a functional
# Metal/CUDA backend.

_metal_ok() = try
    @eval using Metal
    Metal.functional()
catch
    false
end

_cuda_ok() = try
    Base.find_package("CUDA") === nothing && return false
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Small diblock WLC melt with nonzero chi (segregating, non-uniform profile -- exercises
# the position-varying bend/translate step order, not just a uniform-melt fixed point).
function _wlc_gpu_test_system(device)
    b, lp = [1.0, 1.0], [3.0, 3.0]
    chi = zeros(2, 2); chi[1, 2] = chi[2, 1] = 12.0
    model = ClassicalDFT.SCFTWormLikeChainFluid([("diblock", ["A"=>4, "B"=>4])], b, lp, chi;
                                                 rho0=1.0, kappa=20.0, L_max=6)
    mol_structure = Dict("diblock" => ClassicalDFT.custom_structure("A"^4 * "B"^4))
    structure = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, 8.0], 24)
    return ClassicalDFT.SCFTSystem(model, structure, ClassicalDFT.DFTOptions(device);
        mol_structure=mol_structure, ensemble=[:canonical], n_molecules=[3.0])
end

function _run_wlc_gpu_case(device)
    system = _wlc_gpu_test_system(device)
    ρ = ClassicalDFT.initialize_profiles(system)
    ClassicalDFT.converge!(system, ρ; verbose=false, tol=1e-4, maxit=300)

    w = similar(ρ)
    w_bulk = ClassicalDFT.compute_bulk_fields(system.model, ClassicalDFT.compute_bulk_densities(system))
    ClassicalDFT.compute_fields!(system, ρ, w)
    cache = ClassicalDFT.preallocate_propagator(system, system.propagator, ρ, device)
    ClassicalDFT.propagate!(system, ρ, w, cache; w_bulk=w_bulk)
    q_in, q_out = ClassicalDFT.cache_q_in(cache), ClassicalDFT.cache_q_out(cache)
    S = ClassicalDFT.orientation_order_parameter(system, q_in, q_out; axis=1)
    return Array(ρ), Array(S)
end

@testset verbose = true "WLC GPU backend parity" begin

if _metal_ok()
    @testset "Metal vs CPU: density + S_A parity" begin
        ρ_cpu, S_cpu = _run_wlc_gpu_case(ClassicalDFT.CPU())
        ρ_mtl, S_mtl = _run_wlc_gpu_case(Metal.MetalBackend())
        @test size(ρ_cpu) == size(ρ_mtl)
        @test all(isfinite, ρ_mtl)
        @test all(isfinite, S_mtl)
        @test isapprox(ρ_cpu, ρ_mtl; rtol=0.02, atol=1e-3)
        @test isapprox(S_cpu, S_mtl; rtol=0.05, atol=1e-2)
    end
else
    @info "Skipping WLC Metal parity test: Metal not functional on this machine"
end

# CUDA correctness for WLC is UNVERIFIED on this machine (no CUDA hardware available) --
# this test is written defensively so it is ready to run in CI or on a CUDA machine, but
# passing here (by skipping) is NOT evidence of CUDA correctness. Confirm on real CUDA
# hardware before considering CUDA support for WLCPropagator validated.
if _cuda_ok()
    @testset "WLC CUDA vs CPU parity (UNVERIFIED locally)" begin
        ρ_cpu, S_cpu = _run_wlc_gpu_case(ClassicalDFT.CPU())
        ρ_cu, S_cu = _run_wlc_gpu_case(CUDA.CUDABackend())
        @test size(ρ_cpu) == size(ρ_cu)
        @test isapprox(ρ_cpu, ρ_cu; rtol=1e-3, atol=1e-6)
        @test isapprox(S_cpu, S_cu; rtol=1e-3, atol=1e-6)
    end
else
    @info "Skipping WLC CUDA parity test: CUDA.jl not installed/functional in this environment"
end

end
