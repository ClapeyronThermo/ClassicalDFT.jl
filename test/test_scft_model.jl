using Test, ClassicalDFT, Clapeyron

# Unit tests for the modular SCFTModel API (chains × interactions).
@testset "SCFTModel" begin
    grouplist = [("diblock", ["A"=>4, "B"=>4]), ("solv", ["S"=>1])]
    chi = [0.0 1.0 0.5; 1.0 0.0 0.2; 0.5 0.2 0.0]

    @testset "chain specification forms" begin
        m1 = SCFTModel(grouplist; rho0=1.0, chain=GaussianChain(b=1.3))
        @test ClassicalDFT.segment_lengths(m1) == [1.3, 1.3, 1.3]
        m2 = SCFTModel(grouplist; rho0=1.0,
                       chain=Dict("A"=>RigidRod(b=1.0), "B"=>WormLikeChain(b=1.0, lp=4.0), "S"=>WormLikeChain(b=0.5, lp=1.0)))
        @test ClassicalDFT.persistence_lengths(m2) == [Inf, 4.0, 1.0]
        @test ClassicalDFT.has_orientation(m2)
        m3 = SCFTModel(grouplist; rho0=1.0, chain=[GaussianChain(b=1.0), GaussianChain(b=2.0), GaussianChain(b=3.0)])
        @test ClassicalDFT.segment_lengths(m3) == [1.0, 2.0, 3.0]
        @test !ClassicalDFT.has_orientation(m3)
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, chain=Dict("A"=>GaussianChain()))
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, chain=[GaussianChain()])
    end

    @testset "species-named interaction parameters" begin
        named = Dict(("A","B") => 1.0, ("A","S") => 0.5, ("B"=>"S") => 0.2)
        m = SCFTModel(grouplist; rho0=1.0, interactions=(FloryHugginsChi(named),))
        @test ClassicalDFT.get_interaction(m, FloryHugginsChi).chi == chi
        @test ClassicalDFT.get_interaction(m, MaierSaupe) === nothing
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, interactions=(FloryHugginsChi(Dict(("A","Q")=>1.0)),))
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, interactions=(FloryHugginsChi([0.0 1.0; 2.0 0.0]),))
    end

    @testset "Maier-Saupe needs orientation-resolved species" begin
        nu = zeros(3, 3); nu[1, 1] = 2.0
        @test_throws ArgumentError SCFTModel(grouplist; rho0=1.0, chain=GaussianChain(),
                                             interactions=(MaierSaupe(nu),))
        m = SCFTModel(grouplist; rho0=1.0,
                      chain=Dict("A"=>RigidRod(), "B"=>GaussianChain(), "S"=>GaussianChain()),
                      interactions=(MaierSaupe(nu),))
        @test ClassicalDFT.has_maier_saupe(m)
    end

    @testset "legacy constructors build the equivalent SCFTModel" begin
        old = SCFTLatticeFluid(grouplist, [1.0, 1.2, 0.8], chi; rho0=1.0, kappa=20.0)
        @test old isa SCFTModel
        @test ClassicalDFT.segment_lengths(old) == [1.0, 1.2, 0.8]
        @test ClassicalDFT.get_interaction(old, HelfandCompressibility).kappa == 20.0
        @test ClassicalDFT.get_interaction(old, FloryHugginsChi).chi == chi
        wlc = SCFTWormLikeChainFluid(grouplist, [1.0, 1.0, 1.0], [3.0, Inf, 1.0], chi;
                                     rho0=1.0, kappa=20.0, nu=[1.0 0 0; 0 0 0; 0 0 0], L_max=5)
        @test wlc.L_max == 5 && ClassicalDFT.has_maier_saupe(wlc)
    end

    @testset "bulk: AD fields match closed form; a_res gives segment-summed μ" begin
        κ, ρ0 = 15.0, 1.3
        m = SCFTModel(grouplist; rho0=ρ0, interactions=(FloryHugginsChi(chi), HelfandCompressibility(κ)))
        ρ = [0.4, 0.5, 0.3]
        w = ClassicalDFT.compute_bulk_fields(m, ρ)
        w_ref = chi * ρ ./ ρ0 .+ (κ / ρ0) * (sum(ρ) / ρ0 - 1)
        @test w ≈ w_ref rtol = 1e-12
        # μ_res,i/(R̄T) = Σ_α n_iα w_α  (molecule i's residual μ is the sum over its segments)
        V, T = 2.0, 300.0
        z = [0.1, 0.25]
        nseg = [sum(z[i] .* m.groups.n_flattenedgroups[i] for i in eachindex(z))...] ./ V
        wz = ClassicalDFT.compute_bulk_fields(m, nseg)
        μ = Clapeyron.VT_chemical_potential_res(m, V, T, z) ./ (Clapeyron.R̄ * T)
        @test μ ≈ [sum(m.groups.n_flattenedgroups[i] .* wz) for i in eachindex(z)] rtol = 1e-10
    end

    @testset "interaction order does not change the solution" begin
        function solve(ints)
            m = SCFTModel([("diblock", ["A"=>5, "B"=>5])]; rho0=1.0, chain=GaussianChain(b=1.0), interactions=ints)
            L, ng = 5.0, 33
            st = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng)
            sys = ClassicalDFT.SCFTSystem(m, st, ClassicalDFT.DFTOptions();
                mol_structure=Dict("diblock" => ClassicalDFT.custom_structure("AAAAABBBBB")),
                ensemble=[:canonical], n_molecules=[L / 10])
            ρ = ClassicalDFT.initialize_profiles(sys)
            z = range(0, L, length=ng)
            ρ[:, 1] .+= 0.05 .* cos.(2π .* z ./ L); ρ[:, 2] .-= 0.05 .* cos.(2π .* z ./ L)
            ClassicalDFT.converge!(sys, ρ; verbose=false, tol=1e-10, maxit=2000)
            return ρ
        end
        fh = FloryHugginsChi([0.0 3.0; 3.0 0.0]); hf = HelfandCompressibility(20.0)
        ρa, ρb = solve((fh, hf)), solve((hf, fh))
        @test maximum(ρa[:, 1]) - minimum(ρa[:, 1]) > 0.3   # actually ordered
        @test ρa ≈ ρb rtol = 1e-8
    end

    @testset "mixed Gaussian/orientation chains need the composite propagator" begin
        m = SCFTModel([("rc", ["A"=>3, "B"=>3])]; rho0=1.0,
                      chain=Dict("A"=>RigidRod(), "B"=>GaussianChain()),
                      interactions=(HelfandCompressibility(20.0),))
        st = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, 6.0], 16)
        @test_throws ErrorException ClassicalDFT.SCFTSystem(m, st, ClassicalDFT.DFTOptions();
            mol_structure=Dict("rc" => ClassicalDFT.custom_structure("AAABBB")),
            ensemble=[:canonical], n_molecules=[1.0])
    end
end
