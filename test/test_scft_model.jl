@isdefined(SCFT_TEST_UTILS) || include("scft_test_utils.jl")

# SCFTModel construction (chains × interactions) and the physics of each interaction.
@testset "SCFTModel" begin
    grouplist = [("diblock", ["A"=>4, "B"=>4]), ("solv", ["S"=>1])]
    chi3 = [0.0 1.0 0.5; 1.0 0.0 0.2; 0.5 0.2 0.0]

    @testset "chain specification forms" begin
        m1 = SCFTModel(grouplist; rho0=1.0, chain=GaussianChain(b=1.3))
        @test CDFT.segment_lengths(m1) == [1.3, 1.3, 1.3]
        m2 = SCFTModel(grouplist; rho0=1.0,
                       chain=Dict("A"=>RigidRod(b=1.0), "B"=>WormLikeChain(b=1.0, lp=4.0), "S"=>WormLikeChain(b=0.5, lp=1.0)))
        @test CDFT.persistence_lengths(m2) == [Inf, 4.0, 1.0]
        @test CDFT.has_orientation(m2)
        m3 = SCFTModel(grouplist; rho0=1.0, chain=[GaussianChain(b=1.0), GaussianChain(b=2.0), GaussianChain(b=3.0)])
        @test CDFT.segment_lengths(m3) == [1.0, 2.0, 3.0]
        @test !CDFT.has_orientation(m3)
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, chain=Dict("A"=>GaussianChain()))
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, chain=[GaussianChain()])
    end

    @testset "species-named parameters and validation" begin
        named = Dict(("A","B") => 1.0, ("A","S") => 0.5, ("B"=>"S") => 0.2)
        m = SCFTModel(grouplist; rho0=1.0, interactions=(FloryHugginsChi(named),))
        @test CDFT.get_interaction(m, FloryHugginsChi).chi == chi3
        @test CDFT.get_interaction(m, MaierSaupe) === nothing
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, interactions=(FloryHugginsChi(Dict(("A","Q")=>1.0)),))
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, interactions=(FloryHugginsChi([0.0 1.0; 2.0 0.0]),))
        # Maier-Saupe needs orientation-resolved species
        nu = zeros(3, 3); nu[1, 1] = 2.0
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, interactions=(MaierSaupe(nu),))
        @test CDFT.has_maier_saupe(SCFTModel(grouplist; rho0=1.0, interactions=(MaierSaupe(nu),),
            chain=Dict("A"=>RigidRod(), "B"=>GaussianChain(), "S"=>GaussianChain())))
        # one total-density constraint at a time
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0,
            interactions=(Incompressible(), HelfandCompressibility(10.0)))
    end

    @testset "legacy constructors build the equivalent SCFTModel" begin
        old = SCFTLatticeFluid(grouplist, [1.0, 1.2, 0.8], chi3; rho0=1.0, kappa=20.0)
        @test old isa SCFTModel
        @test CDFT.segment_lengths(old) == [1.0, 1.2, 0.8]
        @test CDFT.get_interaction(old, HelfandCompressibility).kappa == 20.0
        @test CDFT.get_interaction(old, FloryHugginsChi).chi == chi3
        wlc = SCFTWormLikeChainFluid(grouplist, [1.0, 1.0, 1.0], [3.0, Inf, 1.0], chi3;
                                     rho0=1.0, kappa=20.0, nu=[1.0 0 0; 0 0 0; 0 0 0], L_max=5)
        @test wlc.L_max == 5 && CDFT.has_maier_saupe(wlc)
    end

    @testset "bulk: AD fields match closed form; a_res gives segment-summed μ" begin
        κ, ρ0 = 15.0, 1.3
        m = SCFTModel(grouplist; rho0=ρ0, interactions=(FloryHugginsChi(chi3), HelfandCompressibility(κ)))
        ρ = [0.4, 0.5, 0.3]
        @test CDFT.compute_bulk_fields(m, ρ) ≈ chi3 * ρ ./ ρ0 .+ (κ / ρ0) * (sum(ρ) / ρ0 - 1) rtol = 1e-12
        # μ_res,i/(R̄T) = Σ_α n_iα w_α: molecule i's residual μ is the sum over its segments
        V, T, z = 2.0, 300.0, [0.1, 0.25]
        nseg = sum(z[i] .* m.groups.n_flattenedgroups[i] for i in eachindex(z)) ./ V
        wz = CDFT.compute_bulk_fields(m, nseg)
        μ = Clapeyron.VT_chemical_potential_res(m, V, T, z) ./ (Clapeyron.R̄ * T)
        @test μ ≈ [sum(m.groups.n_flattenedgroups[i] .* wz) for i in eachindex(z)] rtol = 1e-10
    end

    @testset "propagator selection" begin
        @test scft_system("AAABBB").propagator isa CDFT.DiscreteGaussianChainPropagator
        sys = scft_system("AAABBB"; chain=Dict("A"=>RigidRod(), "B"=>GaussianChain()), L_max=4)
        @test sys.propagator isa CDFT.WLCPropagator
        @test sort(collect(keys(sys.propagator.gauss_kernel))) == [(1, 2), (2, 2)]
        @test collect(keys(sys.propagator.bend_eig)) == [(1, 1)]
    end

    @testset "interaction order does not change the solution" begin
        fh, hf = FloryHugginsChi([0.0 3.0; 3.0 0.0]), HelfandCompressibility(20.0)
        ρa, _ = solve_lamella(scft_system("AAAAABBBBB"; interactions=(fh, hf), L=5.0, ng=33))
        ρb, _ = solve_lamella(scft_system("AAAAABBBBB"; interactions=(hf, fh), L=5.0, ng=33))
        @test contrast(ρa) > 0.3
        @test ρa ≈ ρb rtol = 1e-8
    end

    # Interaction energy (beyond -∫wρ) exactly as free_energy evaluates it.
    function energy(sys, int, ρ)
        dz = CDFT.structure_dz(sys.structure)
        per_∫(f) = sum(f) * prod(dz)
        ctx = (; rho0 = sys.model.rho0, nd = 1, per_∫, structure = sys.structure, R_tensor = nothing)
        return CDFT.field_energy(int, ρ, vec(sum(ρ, dims=2)), ctx)
    end
    chi2 = [0.0 2.0; 2.0 0.0]
    scalar_interactions = [
        "FloryHugginsChi"        => FloryHugginsChi(chi2),
        "SmearedFloryHugginsChi" => SmearedFloryHugginsChi(chi2, 0.4),
        "HelfandCompressibility" => HelfandCompressibility(20.0),
        "EdwardsExcludedVolume"  => EdwardsExcludedVolume([0.7 0.3; 0.3 1.1]),
    ]

    @testset "$name: field = δU/δρ and bulk field = uniform field" for (name, int) in scalar_interactions
        sys = scft_system("AAAAABBBBB"; interactions=(int,), L=5.0, ng=24)
        bound = sys.model.interactions[1]
        ρ = 0.5 .+ 0.2 .* rand(MersenneTwister(1), 24, 2)
        w = zero(ρ)
        CDFT.compute_fields!(sys, ρ, w)
        dz, ε = CDFT.structure_dz(sys.structure)[1], 1e-6
        for (i, α) in ((3, 1), (11, 2), (20, 1))
            ρp = copy(ρ); ρp[i, α] += ε
            ρm = copy(ρ); ρm[i, α] -= ε
            @test (energy(sys, bound, ρp) - energy(sys, bound, ρm)) / (2ε * dz) ≈ w[i, α] rtol = 1e-6 atol = 1e-9
        end
        ρb = [0.37, 0.58]
        wu = zeros(24, 2)
        CDFT.compute_fields!(sys, repeat(ρb', 24, 1), wu)
        @test wu ≈ repeat(CDFT.compute_bulk_fields(sys.model, ρb)', 24, 1) rtol = 1e-10 atol = 1e-12
    end

    @testset "EdwardsExcludedVolume: osmotic pressure of a homopolymer solution" begin
        v, N = 0.8, 20
        m = SCFTModel([("poly", ["A"=>N])]; rho0=1.0, interactions=(EdwardsExcludedVolume(v),))
        V, T, z = 3.0, 300.0, [0.05]
        @test Clapeyron.pressure(m, V, T, z) / (Clapeyron.R̄ * T) ≈ z[1] / V + v * (z[1] * N / V)^2 / 2 rtol = 1e-10
    end

    # Lamella that orders well (same parameters as the regression "lamella_dgc" case).
    chi = [0.0 1.5; 1.5 0.0]
    lam(ints) = scft_system("A"^10 * "B"^10; interactions=ints, L=7.0, ng=65)

    @testset "SmearedFloryHugginsChi: a → 0 recovers local χ; finite a softens the interface" begin
        ρ_local, _ = solve_lamella(lam((FloryHugginsChi(chi), HelfandCompressibility(20.0))); tol=1e-9)
        ρ_small, _ = solve_lamella(lam((SmearedFloryHugginsChi(chi, 1e-4), HelfandCompressibility(20.0))); tol=1e-9)
        ρ_wide, _ = solve_lamella(lam((SmearedFloryHugginsChi(chi, 0.3), HelfandCompressibility(20.0))); tol=1e-9)
        @test contrast(ρ_local) > 0.5
        @test ρ_small ≈ ρ_local rtol = 1e-6
        @test contrast(ρ_wide) < contrast(ρ_local)
    end

    @testset "Incompressible: Σρ = ρ₀; Helfand approaches it as 1/κ" begin
        ρ_i, r = solve_lamella(lam((FloryHugginsChi(chi), Incompressible(lambda=5.0))); tol=1e-9)
        @test r.converged
        @test max_compressibility_error(ρ_i) < 1e-7
        @test contrast(ρ_i) > 0.5
        # κ·max|Δρ| ≈ 0.38, 0.40 at κ = 10, 20. Larger κ is not used: the Helfand iteration
        # itself stops converging there.
        d10 = maximum(abs.(first(solve_lamella(lam((FloryHugginsChi(chi), HelfandCompressibility(10.0))); tol=1e-9)) .- ρ_i))
        d20 = maximum(abs.(first(solve_lamella(lam((FloryHugginsChi(chi), HelfandCompressibility(20.0))); tol=1e-9)) .- ρ_i))
        @test d20 < d10
        @test 20 * d20 ≈ 10 * d10 rtol = 0.15
    end
end
