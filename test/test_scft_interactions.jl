using Test, ClassicalDFT, Clapeyron, Random

# Physics checks for the SCFT interaction submodels.
const _CHI2 = [0.0 2.0; 2.0 0.0]

function _diblock_system(ints; N=10, L=5.0, ng=33, b=1.0)
    m = SCFTModel([("diblock", ["A"=>N÷2, "B"=>N÷2])]; rho0=1.0, chain=GaussianChain(b=b), interactions=ints)
    st = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng)
    return ClassicalDFT.SCFTSystem(m, st, ClassicalDFT.DFTOptions();
        mol_structure=Dict("diblock" => ClassicalDFT.custom_structure("A"^(N÷2) * "B"^(N÷2))),
        ensemble=[:canonical], n_molecules=[L / N])
end

function _solve!(sys; amp=0.05, tol=1e-10, maxit=3000)
    ρ = ClassicalDFT.initialize_profiles(sys)
    L = sys.structure.ngrid[1] * ClassicalDFT.structure_dz(sys.structure)[1]
    z = range(0, L, length=sys.structure.ngrid[1])
    ρ[:, 1] .+= amp .* cos.(2π .* z ./ L)
    ρ[:, 2] .-= amp .* cos.(2π .* z ./ L)
    ClassicalDFT.converge!(sys, ρ; verbose=false, tol=tol, maxit=maxit)
    return ρ
end

# Interaction energy (beyond -∫wρ) of a single interaction, as free_energy evaluates it.
function _energy(sys, int, ρ)
    nd = 1
    dz = ClassicalDFT.structure_dz(sys.structure)
    per_∫(f) = sum(f) * prod(dz)
    ρt = vec(sum(ρ, dims=2))
    ctx = (; rho0 = sys.model.rho0, nd, per_∫, structure = sys.structure, R_tensor = nothing)
    return ClassicalDFT.field_energy(int, ρ, ρt, ctx)
end

@testset "SCFT interactions" begin
    ints_under_test = [
        "FloryHugginsChi"        => FloryHugginsChi(_CHI2),
        "SmearedFloryHugginsChi" => SmearedFloryHugginsChi(_CHI2, 0.4),
        "HelfandCompressibility" => HelfandCompressibility(20.0),
        "EdwardsExcludedVolume"  => EdwardsExcludedVolume([0.7 0.3; 0.3 1.1]),
    ]

    @testset "field = δU/δρ ($name)" for (name, int) in ints_under_test
        sys = _diblock_system((int,); ng=24)
        bound = sys.model.interactions[1]
        rng = MersenneTwister(1)
        ρ = 0.5 .+ 0.2 .* rand(rng, 24, 2)
        w = zero(ρ)
        ClassicalDFT.compute_fields!(sys, ρ, w)
        dz = ClassicalDFT.structure_dz(sys.structure)[1]
        ε = 1e-6
        for (i, α) in ((3, 1), (11, 2), (20, 1))
            ρp = copy(ρ); ρp[i, α] += ε
            ρm = copy(ρ); ρm[i, α] -= ε
            dU = (_energy(sys, bound, ρp) - _energy(sys, bound, ρm)) / (2ε * dz)
            @test dU ≈ w[i, α] rtol = 1e-6 atol = 1e-9
        end
    end

    @testset "bulk fields = field of a uniform profile ($name)" for (name, int) in ints_under_test
        sys = _diblock_system((int,); ng=16)
        ρb = [0.37, 0.58]
        ρ = repeat(ρb', 16, 1)
        w = zero(ρ)
        ClassicalDFT.compute_fields!(sys, ρ, w)
        wb = ClassicalDFT.compute_bulk_fields(sys.model, ρb)
        @test w ≈ repeat(wb', 16, 1) rtol = 1e-10 atol = 1e-12
    end

    @testset "Edwards: osmotic pressure of a homopolymer solution" begin
        v, N = 0.8, 20
        m = SCFTModel([("poly", ["A"=>N])]; rho0=1.0, interactions=(EdwardsExcludedVolume(v),))
        V, T, z = 3.0, 300.0, [0.05]
        ρseg = z[1] * N / V
        p = Clapeyron.pressure(m, V, T, z)
        @test p / (Clapeyron.R̄ * T) ≈ z[1] / V + v * ρseg^2 / 2 rtol = 1e-10
    end

    # Lamella that orders well (same parameters as the regression "lamella_dgc" case).
    chi = [0.0 1.5; 1.5 0.0]
    lam(ints) = _diblock_system(ints; N=20, L=7.0, ng=65)

    @testset "SmearedFloryHugginsChi: a → 0 recovers local χ" begin
        ρ_local = _solve!(lam((FloryHugginsChi(chi), HelfandCompressibility(20.0))); tol=1e-9)
        ρ_small = _solve!(lam((SmearedFloryHugginsChi(chi, 1e-4), HelfandCompressibility(20.0))); tol=1e-9)
        @test maximum(ρ_local[:, 1]) - minimum(ρ_local[:, 1]) > 0.5
        @test ρ_small ≈ ρ_local rtol = 1e-6
        # a finite range softens the interface: smaller composition contrast
        ρ_wide = _solve!(lam((SmearedFloryHugginsChi(chi, 0.3), HelfandCompressibility(20.0))); tol=1e-9)
        @test maximum(ρ_wide[:, 1]) - minimum(ρ_wide[:, 1]) < maximum(ρ_local[:, 1]) - minimum(ρ_local[:, 1])
    end

    @testset "Incompressible: Σρ = ρ₀; Helfand approaches it as 1/κ" begin
        @test_throws ArgumentError SCFTModel([("x", ["A"=>1])]; rho0=1.0,
            interactions=(Incompressible(), HelfandCompressibility(10.0)))
        ρ_i = _solve!(lam((FloryHugginsChi(chi), Incompressible(lambda=5.0))); tol=1e-9)
        @test !any(isnan, ρ_i)
        @test maximum(abs.(sum(ρ_i, dims=2) .- 1.0)) < 1e-7
        @test maximum(ρ_i[:, 1]) - minimum(ρ_i[:, 1]) > 0.5
        # Soft Helfand penalty → strict incompressibility with error ∝ 1/κ (κ·Δ ≈ 0.38, 0.40
        # at κ = 10, 20). Larger κ is not used: the Helfand iteration itself stops converging.
        d10 = maximum(abs.(_solve!(lam((FloryHugginsChi(chi), HelfandCompressibility(10.0))); tol=1e-9) .- ρ_i))
        d20 = maximum(abs.(_solve!(lam((FloryHugginsChi(chi), HelfandCompressibility(20.0))); tol=1e-9) .- ρ_i))
        @test d20 < d10
        @test 20 * d20 ≈ 10 * d10 rtol = 0.15
    end
end
