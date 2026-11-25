@isdefined(SCFT_TEST_UTILS) || include("scft_test_utils.jl")

# Orientational physics: the Maier-Saupe SCFT solver and the nematic bulk thermodynamics
# (src/models/SCFT/nematic.jl). A rigid rod of N beads shares one orientation, so its
# Maier-Saupe coupling is a = Nν (isotropic-nematic transition at a ≈ 4.49).
const _Ref = RodMeltReference

rod_melt(ν; N=6, L_max=6, κ=20.0, ng=16) = scft_system("A"^N; chain=RigidRod(b=1.0), L=6.0, ng, L_max,
    interactions=(HelfandCompressibility(κ), MaierSaupe(fill(ν, 1, 1))))

# Converge a uniform system seeded nematic along z; returns (H/V, segment densities, S, sys).
function uniform_nematic(seq, chain, ints; director=(0, 0, 1))
    sys = scft_system(seq; chain, interactions=ints, L=6.0, ng=8, L_max=8)
    ρ = CDFT.initialize_profiles(sys)
    r = CDFT.converge!(sys, ρ; tol=1e-11, maxit=3000, director, nematic_order=0.6)
    @test r.converged
    H = CDFT.free_energy(sys, ρ, r.w, r.Q; R_tensor=r.R_tensor)
    return H / box_length(sys), vec(sum(ρ, dims=1)) ./ size(ρ, 1), uniaxial_order(r.R_tensor, ρ), sys
end

@testset "SCFT orientation" begin
    @testset "Maier-Saupe solver converges where the old damped update failed (ν = 2, χ = $χ)" for χ in (1.2, 2.0)
        sys = scft_system("A"^6 * "B"^6; L=12.0, ng=48,
            chain=Dict("A" => WormLikeChain(b=1.0, lp=8.0), "B" => WormLikeChain(b=1.0, lp=1.0)),
            interactions=(HelfandCompressibility(20.0), FloryHugginsChi([0.0 χ; χ 0.0]), MaierSaupe([2.0 0.0; 0.0 0.0])))
        ρ, r = solve_lamella(sys; tol=1e-8, maxit=4000)
        @test r.converged && !any(isnan, ρ)
        @test contrast(ρ) > 0.5
        @test max_compressibility_error(ρ) < 0.1     # soft Helfand κ = 20
    end

    @testset "SCFT rod melt: isotropic below the transition, nematic above, theory in between" begin
        # ν = 0.5 (a = 3): an aligned seed relaxes back to isotropic
        sys = rod_melt(0.5); ρ = CDFT.initialize_profiles(sys)
        r = CDFT.converge!(sys, ρ; tol=1e-9, maxit=2000, director=(0, 0, 1), nematic_order=0.5)
        @test r.converged && abs(uniaxial_order(r.R_tensor, ρ)[1]) < 1e-6
        # ν = 6: an aligned seed grows into a nematic, which must be reported as such
        sys = rod_melt(6.0); ρ = CDFT.initialize_profiles(sys)
        r = CDFT.converge!(sys, ρ; tol=1e-9, maxit=2000, director=(0, 0, 1), nematic_order=0.1)
        @test r.converged && uniaxial_order(r.R_tensor, ρ)[1] > 0.5
        # without a seed the isotropic state is an exact fixed point
        sys = rod_melt(6.0); ρ = CDFT.initialize_profiles(sys)
        r = CDFT.converge!(sys, ρ; tol=1e-9, maxit=2000)
        @test r.converged && abs(uniaxial_order(r.R_tensor, ρ)[1]) < 1e-8
        # ν = 1 (a = 6): S matches classical Maier-Saupe within the L_max = 8 truncation
        sys = rod_melt(1.0; L_max=8); ρ = CDFT.initialize_profiles(sys)
        r = CDFT.converge!(sys, ρ; tol=1e-10, maxit=3000, director=(0, 0, 1), nematic_order=0.5)
        @test r.converged
        @test uniaxial_order(r.R_tensor, ρ)[1] ≈ _Ref.S_self(6.0) atol = 1e-3
        # director seed is normalized and validated
        @test CDFT.SCFTProblem(sys; director=(0, 0, 2)).director == (0.0, 0.0, 1.0)
        @test_throws ArgumentError CDFT.SCFTProblem(sys; director=(0, 0, 0))
    end

    @testset "isotropic Maier-Saupe term: uniform SCFT free energy = bulk free energy" begin
        nu = [0.4 0.2; 0.2 0.3]
        HV, ρseg, S, sys = uniform_nematic("AAABBB", WormLikeChain(b=1.0, lp=2.0),
            (FloryHugginsChi([0.0 0.5; 0.5 0.0]), HelfandCompressibility(20.0), MaierSaupe(nu)); director=nothing)
        @test all(abs.(S) .< 1e-10)
        @test HV ≈ bulk_free_energy_density(sys.model, ρseg) rtol = 1e-10
    end

    @testset "bulk rigid rod vs classical Maier-Saupe (ν = $ν)" for ν in (1.0, 2.0)
        nb = NematicBulk(rod_melt(ν; L_max=12))
        r = bulk_order_parameter(nb, [1.0 / 6])
        a, S_ref = 6ν, _Ref.S_self(6ν)
        @test r.converged
        @test r.S[1] ≈ S_ref atol = 1e-3                                       # L_max = 12 truncation
        @test 6 * r.Δf ≈ a * S_ref^2 / 2 - _Ref.logZ(a * S_ref) atol = 5e-3     # per rod
        lo = bulk_order_parameter(NematicBulk(rod_melt(0.5)), [1.0 / 6])        # a = 3: isotropic only
        @test lo.converged && abs(lo.S[1]) < 1e-8 && abs(lo.Δf) < 1e-10
    end

    @testset "bulk free energy = uniform nematic SCFT free energy ($label)" for (label, seq, chain, ints) in (
            ("rod melt", "AAAAAA", RigidRod(b=1.0), (HelfandCompressibility(20.0), MaierSaupe(fill(1.0, 1, 1)))),
            ("rod-coil", "AAAABBBB", Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0)),
             (FloryHugginsChi(Dict(("A","B") => 0.3)), HelfandCompressibility(20.0), MaierSaupe(Dict(("A","A") => 4.0)))),
            ("branched WLC/Gaussian", "AA(BB)(BBB)A", Dict("A" => WormLikeChain(b=1.0, lp=6.0), "B" => GaussianChain(b=1.0)),
             (HelfandCompressibility(20.0), MaierSaupe(Dict(("A","A") => 25.0)))))
        HV, ρseg, S_scft, sys = uniform_nematic(seq, chain, ints)
        b = bulk_order_parameter(NematicBulk(sys), [1.0 / count(isletter, seq)])
        @test S_scft[1] > 0.5                                    # genuinely nematic
        @test b.S ≈ S_scft atol = 1e-7
        @test HV ≈ bulk_free_energy_density(sys.model, ρseg) + b.Δf rtol = 1e-8
    end

    @testset "rod melt I–N coexistence vs independent calculation" begin
        N, κ, ν = 6, 5.0, 0.76
        r = isotropic_nematic_coexistence(NematicBulk(rod_melt(ν; κ, L_max=12)), [0.98 / N], [1.02 / N])
        (ρI, ρN), S_ref = _Ref.coexistence(0.98 / N, 1.02 / N; N, κ, ν, ρ0=1.0)
        @test r.converged
        @test r.ρ_iso[1] ≈ ρI rtol = 1e-8
        @test r.ρ_nem[1] ≈ ρN rtol = 1e-8
        @test r.S_nem[1] ≈ S_ref atol = 1e-6
        @test r.ρ_nem[1] > r.ρ_iso[1]                            # nematic is the denser phase
    end

    @testset "rods + Gaussian solvent: coexistence at fixed isotropic composition" begin
        m = SCFTModel([("rod", ["A"=>6]), ("solv", ["S"=>1])]; rho0=1.0, L_max=12,
                      chain=Dict("A" => RigidRod(b=1.0), "S" => GaussianChain(b=1.0)),
                      interactions=(HelfandCompressibility(10.0), MaierSaupe(Dict(("A","A") => 1.2))))
        nb = NematicBulk(m; mol_structure=Dict("rod" => CDFT.custom_structure("AAAAAA"), "solv" => CDFT.custom_structure("S")))
        φI = 0.65
        r = isotropic_nematic_coexistence(nb, [φI / 6, 1 - φI], [0.8 / 6, 0.2])
        φ(ρ) = 6ρ[1] / (6ρ[1] + ρ[2])
        @test r.converged
        @test bulk_chemical_potential(nb, r.ρ_iso, zeros(2)) ≈ bulk_chemical_potential(nb, r.ρ_nem, r.S_nem) atol = 1e-8
        @test bulk_pressure(nb, r.ρ_iso, zeros(2)) ≈ bulk_pressure(nb, r.ρ_nem, r.S_nem) atol = 1e-8
        @test φ(r.ρ_iso) ≈ φI rtol = 1e-12                       # composition held fixed
        @test φ(r.ρ_nem) > φI                                     # nematic enriched in rods
        @test r.S_nem[2] == 0                                     # Gaussian solvent
    end

    @testset "no orientation-resolved species: always isotropic" begin
        r = bulk_order_parameter(NematicBulk(scft_system("AAA")), [1 / 3])
        @test r.converged && r.S == [0.0] && abs(r.Δf) < 1e-12
    end
end
