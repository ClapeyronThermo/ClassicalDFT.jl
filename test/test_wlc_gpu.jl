using Test, ClassicalDFT
using ClassicalDFT: SCFTModel, GaussianChain, WormLikeChain, RigidRod, FloryHugginsChi,
    SmearedFloryHugginsChi, HelfandCompressibility, Incompressible, MaierSaupe

# Backend-parity tests for the WLC propagator's KernelAbstractions support -- checks that
# a converged density profile AND orientation_order_parameter's output agree between a
# GPU backend and CPU, run on the SAME model/structure. Self-skips (registering no testset
# at all, not a spurious skipped/broken entry) on any machine without a functional
# Metal/CUDA backend.

# `Metal.functional()` must run via `@eval`: a plain call in the same function would
# execute in the world before `using Metal`, throw, and silently skip the suite.
_metal_ok() = try
    @eval using Metal
    @eval Metal.functional()
catch
    false
end

_cuda_ok() = try
    Base.find_package("CUDA") === nothing && return false
    @eval using CUDA
    @eval CUDA.functional()
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

    S = ClassicalDFT.orientation_order_parameter(system, ρ; axis=1)
    return Array(ρ), Array(S)
end

# Parity cases for the modular SCFTModel paths (composite and branched chains, the new
# interactions, Maier-Saupe). Each builder takes a device and returns a 1D system.
function _modular_gpu_system(device, case::Symbol)
    st = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, 8.0], 24)
    opts = ClassicalDFT.DFTOptions(device)
    chi = Dict(("A", "B") => 2.5)
    build(gl, seq, chain, ints; L_max=6) = ClassicalDFT.SCFTSystem(
        SCFTModel(gl; rho0=1.0, chain=chain, interactions=ints, L_max=L_max), st, opts;
        mol_structure=Dict(gl[1][1] => ClassicalDFT.custom_structure(seq)),
        ensemble=[:canonical], n_molecules=[8.0 / count(isletter, seq)])
    if case === :rodcoil
        return build([("rc", ["A"=>4, "B"=>6])], "AAAABBBBBB",
            Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0)),
            (FloryHugginsChi(chi), HelfandCompressibility(20.0)))
    elseif case === :branched
        return build([("star", ["A"=>3, "B"=>5])], "AA(BB)(BBB)A",
            Dict("A" => WormLikeChain(b=1.0, lp=3.0), "B" => GaussianChain(b=1.0)),
            (FloryHugginsChi(chi), HelfandCompressibility(20.0)))
    elseif case === :smeared
        return build([("db", ["A"=>5, "B"=>5])], "AAAAABBBBB", GaussianChain(b=1.0),
            (SmearedFloryHugginsChi(chi, 0.3), HelfandCompressibility(20.0)))
    elseif case === :incompressible
        return build([("db", ["A"=>5, "B"=>5])], "AAAAABBBBB", GaussianChain(b=1.0),
            (FloryHugginsChi(chi), Incompressible(lambda=5.0)))
    elseif case === :maiersaupe
        return build([("db", ["A"=>4, "B"=>4])], "AAAABBBB", WormLikeChain(b=1.0, lp=3.0),
            (FloryHugginsChi(chi), HelfandCompressibility(20.0), MaierSaupe(Dict(("A", "A") => 0.3))))
    end
    error("unknown case $case")
end

function _run_modular_case(device, case)
    system = _modular_gpu_system(device, case)
    ρ = ClassicalDFT.initialize_profiles(system)
    z = range(0, 8.0, length=24)
    δ = 0.05 .* cos.(2π .* z ./ 8.0)
    ρh = Array(ρ); ρh[:, 1] .+= δ; ρh[:, 2] .-= δ
    copyto!(ρ, ρh)
    # A fixed iteration count compares the same trajectory on both backends, except for
    # Incompressible: its pressure update magnifies early Float32 round-off along the
    # trajectory, so that case is compared at convergence instead.
    if case === :incompressible
        ClassicalDFT.converge!(system, ρ; verbose=false, tol=1e-5, maxit=2000)
    else
        ClassicalDFT.converge!(system, ρ; verbose=false, tol=1e-12, maxit=60)
    end
    return Array(ρ)
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
    @testset "Metal vs CPU: $case" for case in (:rodcoil, :branched, :smeared, :incompressible, :maiersaupe)
        ρ_cpu = _run_modular_case(ClassicalDFT.CPU(), case)
        ρ_mtl = _run_modular_case(Metal.MetalBackend(), case)
        @test all(isfinite, ρ_mtl)
        @test maximum(abs.(ρ_cpu .- ρ_mtl)) < 1e-3   # Metal runs in Float32
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
