@isdefined(SCFT_TEST_UTILS) || include("scft_test_utils.jl")

# Chains mixing Gaussian and orientation-resolved species, linear and branched, through
# WLCPropagator (Gaussian bonds, rod bonds, freely jointed branch points).
@testset "SCFT composite chains" begin
    rng = MersenneTwister(7)
    wlc_path(s) = CDFT.WLCPropagator(s.model, s.species, s.structure, CPU(), Float64; L_max=4)

    @testset "all-Gaussian chain through the WLC path = Gaussian propagator ($seq)" for seq in ("AAABBBB", "AA(BB)(BBB)A")
        chain = Dict("A" => GaussianChain(b=1.0), "B" => GaussianChain(b=1.3))
        sys_g = scft_system(seq; chain)
        sys_w = scft_system(seq; chain, propagator=wlc_path)
        @test sys_g.propagator isa CDFT.DiscreteGaussianChainPropagator
        w = 0.4 .* randn(rng, 24, 2)
        a, b = propagate_fields(sys_g, w), propagate_fields(sys_w, w)
        @test b.Q ≈ a.Q rtol = 1e-12
        @test b.ρ ≈ a.ρ rtol = 1e-12
    end

    rodcoil = Dict("A" => WormLikeChain(b=1.0, lp=4.0), "B" => GaussianChain(b=1.0))
    rodstar = Dict("A" => WormLikeChain(b=1.0, lp=4.0), "B" => RigidRod(b=1.0))
    @testset "uniform field, per-node partition function and mass ($label, $seq)" for (label, chain, seq) in (
            ("rod-coil", rodcoil, "AAAABBBB"), ("rod-coil", rodcoil, "BBAAAABB"), ("rod-coil", rodcoil, "ABBAAB"),
            ("rod-coil", rodcoil, "AA(BB)(BBB)A"), ("rod-coil", rodcoil, "A(B)(B)(B)"),
            ("rod star", rodstar, "AA(BB)(BBB)A"), ("rod star", rodstar, "A(B)(B)(B)"))
        sys = scft_system(seq; chain)
        # uniform field: Q = 1 and bulk densities
        r0 = propagate_fields(sys, zeros(24, 2))
        @test r0.Q[1] ≈ 1.0 rtol = 1e-10
        @test r0.ρ ≈ repeat(CDFT.compute_bulk_densities(sys)', 24, 1) rtol = 1e-10
        # random field: (1/V)∫dr∫du q_in q_out / ef = Q at every node, and ∫ρ_α = n·(#α beads)
        w = 0.5 .* randn(rng, 24, 2)
        r = propagate_fields(sys, w)
        qw = sys.propagator.sht.quad_weight
        dz = CDFT.structure_dz(sys.structure)[1]
        seqv = sys.species.sequence[1]
        for k in eachindex(seqv)
            ef = exp.(r.w_bulk[seqv[k]] .- w[:, seqv[k]])
            qq = selectdim(r.q_in[1], 3, k) .* selectdim(r.q_out[1], 3, k)
            @test sum(CDFT._orientation_marginalize(qq, qw, 2) ./ ef) / 24 ≈ r.Q[1] rtol = 1e-10
        end
        for α in 1:2
            @test sum(r.ρ[:, α]) * dz ≈ sys.species.n_molecules[1] * count(==(α), seqv) rtol = 1e-10
        end
    end

    @testset "rod-coil lamella converges (MaierSaupe on the rod: $(ms))" for ms in (false, true)
        ints = (FloryHugginsChi(Dict(("A", "B") => 2.5)), HelfandCompressibility(20.0))
        ms && (ints = (ints..., MaierSaupe(Dict(("A", "A") => 1.0))))
        sys = scft_system("A"^6 * "B"^10; chain=Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0)),
                          interactions=ints, L=8.0, ng=48)
        ρ, r = solve_lamella(sys; tol=1e-8, maxit=3000)
        @test r.converged
        @test contrast(ρ) > 0.3
        @test max_compressibility_error(ρ) < 0.05
    end
end
