@isdefined(SCFT_TEST_UTILS) || include("scft_test_utils.jl")

# Sequence-defined models (one species per contour node) with any chain/interaction.
sequence_system(model; L=6.0, ng=24) = SCFTSequenceSystem(model, CDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng);
    ensemble=:canonical, n_molecules=L / size(model.groups.flattenedgroups, 1))

@testset "SCFT sequence models" begin
    Nc, χ, ν = 8, 2.5, 1.5
    rodcoil_chain = Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0))
    block_ints = (HelfandCompressibility(20.0), FloryHugginsChi(Dict(("A", "B") => χ)), MaierSaupe(Dict(("A", "A") => ν)))
    block = scft_system("AAAABBBB"; chain=rodcoil_chain, interactions=block_ints)

    # Step descriptor f = 0 (rod block) / 1 (coil block); node-pair parameters reproduce the
    # block parameters: χ between blocks, ν between rod nodes.
    d = step_descriptor([0.5, 0.5], [[0.0], [1.0]], Nc)
    model = sequence_model(d; rho0=1.0, gamma=nothing, L_max=6,
        chain = s -> s < 0.5 ? RigidRod(b=1.0) : GaussianChain(b=1.0),
        interactions = (HelfandCompressibility(20.0),
                        FloryHugginsChi(sequence_pair_matrix(d, (f1, f2) -> χ * (f1[1] - f2[1])^2)),
                        MaierSaupe(sequence_pair_matrix(d, (f1, f2) -> ν * (1 - f1[1]) * (1 - f2[1])))))
    seq = sequence_system(model)
    blk(k) = k <= Nc ÷ 2 ? 1 : 2
    to_blocks(ρn) = hcat(sum(ρn[:, 1:Nc÷2], dims=2), sum(ρn[:, Nc÷2+1:end], dims=2))

    @testset "construction" begin
        @test length(model.groups.flattenedgroups) == Nc
        @test count(c -> c isa RigidRod, model.chains) == Nc ÷ 2
        @test seq.propagator isa CDFT.WLCPropagator
        @test_throws ArgumentError sequence_model(d; rho0=1.0, chain=[GaussianChain()])   # wrong length
    end

    @testset "step descriptor = block rod-coil model: propagators in a random field" begin
        w_blk = 0.5 .* randn(MersenneTwister(3), 24, 2)
        w_node = hcat((w_blk[:, blk(k)] for k in 1:Nc)...)
        a, b = propagate_fields(block, w_blk), propagate_fields(seq, w_node)
        @test b.Q ≈ a.Q rtol = 1e-10
        @test to_blocks(b.ρ) ≈ a.ρ rtol = 1e-10
    end

    @testset "step descriptor = block rod-coil model: converged lamella and nematic bulk" begin
        ρa, ra = solve_lamella(block; tol=1e-10)
        ρn = CDFT.initialize_profiles(seq)
        L = box_length(seq); z = range(0, L, length=24)
        for k in 1:Nc
            ρn[:, k] .+= (blk(k) == 1 ? 1 : -1) * 0.05 / (Nc ÷ 2) .* cos.(2π .* z ./ L)
        end
        rn = CDFT.converge!(seq, ρn; tol=1e-10, maxit=3000)
        @test ra.converged && rn.converged
        @test to_blocks(ρn) ≈ ρa rtol = 1e-6
        nb_blk, nb_seq = NematicBulk(block), NematicBulk(seq)
        for S0 in (0.0, 0.8)
            @test bulk_order_parameter(nb_seq, [1 / Nc]; S0).Δf ≈ bulk_order_parameter(nb_blk, [1 / Nc]; S0).Δf atol = 1e-9
        end
    end

    @testset "tapered stiffness along the chain" begin
        dt = taper_descriptor(0.0, 1.0, 12)
        m = sequence_model(dt; rho0=1.0, chain = s -> WormLikeChain(b=1.0, lp=1 + 9s),
                           gamma = (f1, f2) -> 300 * sum(abs2, f1 .- f2), L_max=6)
        @test CDFT.persistence_lengths(m) ≈ [1 + 9s for s in dt.s]
        sys = sequence_system(m; L=8.0, ng=32)
        ρ = CDFT.initialize_profiles(sys)
        z = range(0, 8.0, length=32)
        for k in 1:12
            ρ[:, k] .+= (k <= 6 ? 1 : -1) * 0.005 .* cos.(2π .* z ./ 8.0)
        end
        r = CDFT.converge!(sys, ρ; tol=1e-8, maxit=3000)
        @test r.converged && !any(isnan, ρ)
        @test max_compressibility_error(hcat(sum(ρ, dims=2))) < 0.1
    end
end
