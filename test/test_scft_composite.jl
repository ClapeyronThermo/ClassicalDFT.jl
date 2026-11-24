using Test, ClassicalDFT, Clapeyron, Random
using KernelAbstractions: CPU

# Composite (mixed Gaussian / orientation-resolved) chains through WLCPropagator.

function _composite_system(chain, seq; ints=(HelfandCompressibility(20.0),), L=6.0, ng=24, L_max=6,
                           grouplist=nothing, propagator=nothing)
    letters = unique(collect(seq))
    gl = grouplist === nothing ? [("mol", [string(l) => count(==(l), seq) for l in letters])] : grouplist
    m = SCFTModel(gl; rho0=1.0, chain=chain, interactions=ints, L_max=L_max)
    st = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng)
    sys = ClassicalDFT.SCFTSystem(m, st, ClassicalDFT.DFTOptions();
        mol_structure=Dict("mol" => ClassicalDFT.custom_structure(seq)),
        ensemble=[:canonical], n_molecules=[L / count(isletter, seq)])
    propagator === nothing && return sys
    p = propagator(sys)
    return ClassicalDFT.SCFTSystem(sys.model, sys.species, sys.structure, p, sys.options, sys.external_field)
end

# Run the propagators on field w and return (Q, ρ, q_in, q_out, exp_field).
function _propagate(sys, w)
    nd = ClassicalDFT.dimension(sys)
    w_bulk = ClassicalDFT.compute_bulk_fields(sys.model, ClassicalDFT.compute_bulk_densities(sys))
    ρ = zero(w)
    cache = ClassicalDFT.preallocate_propagator(sys, sys.propagator, ρ, CPU())
    ClassicalDFT.propagate!(sys, ρ, w, cache; w_bulk=w_bulk)
    q_in, q_out = ClassicalDFT.cache_q_in(cache), ClassicalDFT.cache_q_out(cache)
    dz = ClassicalDFT.structure_dz(sys.structure)
    Q = ClassicalDFT.compute_partition_functions(sys, w, w_bulk, q_in, dz)
    ClassicalDFT.compute_densities!(sys, w, w_bulk, q_in, q_out, Q, ρ)
    return (; Q, ρ, q_in, q_out, w_bulk)
end

@testset "SCFT composite chains" begin
    rng = MersenneTwister(7)

    @testset "all-Gaussian chain through the composite path = Gaussian propagator" begin
        chain = Dict("A" => GaussianChain(b=1.0), "B" => GaussianChain(b=1.4))
        seq = "AAABBBB"
        sys_g = _composite_system(chain, seq)
        sys_w = _composite_system(chain, seq;
            propagator = s -> ClassicalDFT.WLCPropagator(s.model, s.species, s.structure, CPU(), Float64; L_max=4))
        @test sys_g.propagator isa ClassicalDFT.DiscreteGaussianChainPropagator
        @test sys_w.propagator isa ClassicalDFT.WLCPropagator
        w = 0.4 .* randn(rng, 24, 2)
        a, b = _propagate(sys_g, w), _propagate(sys_w, w)
        @test b.Q ≈ a.Q rtol = 1e-12
        @test b.ρ ≈ a.ρ rtol = 1e-12
    end

    chain_rc = Dict("A" => WormLikeChain(b=1.0, lp=4.0), "B" => GaussianChain(b=1.0))

    @testset "uniform field: Q = 1 and bulk densities ($seq)" for seq in ("AAAABBBB", "BBAAAABB", "ABABAB")
        sys = _composite_system(chain_rc, seq)
        r = _propagate(sys, zeros(24, 2))
        @test r.Q[1] ≈ 1.0 rtol = 1e-10
        bulk = ClassicalDFT.compute_bulk_densities(sys)
        @test r.ρ ≈ repeat(bulk', 24, 1) rtol = 1e-10
    end

    @testset "per-node partition-function identity in a random field ($seq)" for seq in ("AAAABBBB", "BBAAAABB", "ABBAAB")
        sys = _composite_system(chain_rc, seq)
        nd = 1
        w = 0.5 .* randn(rng, 24, 2)
        r = _propagate(sys, w)
        qw = sys.propagator.sht.quad_weight
        dz = ClassicalDFT.structure_dz(sys.structure)[1]
        V = 24 * dz
        seqv = sys.species.sequence[1]
        for k in eachindex(seqv)
            α = seqv[k]
            ef = exp.(r.w_bulk[α] .- w[:, α])
            qq = selectdim(r.q_in[1], nd + 2, k) .* selectdim(r.q_out[1], nd + 2, k)
            Qk = sum(ClassicalDFT._orientation_marginalize(qq, qw, nd + 1) ./ ef) * dz / V
            @test Qk ≈ r.Q[1] rtol = 1e-10
        end
        # mass conservation: ∫ρ_α = n_molecules · (number of α beads)
        n = sys.species.n_molecules[1]
        for α in 1:2
            @test sum(r.ρ[:, α]) * dz ≈ n * count(==(α), seqv) rtol = 1e-10
        end
    end

    @testset "rod-coil lamella converges (with and without Maier-Saupe on the rod)" begin
        chi = Dict(("A", "B") => 2.5)
        for ints in ((FloryHugginsChi(chi), HelfandCompressibility(20.0)),
                     (FloryHugginsChi(chi), HelfandCompressibility(20.0), MaierSaupe(Dict(("A", "A") => 0.3))))
            sys = _composite_system(Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0)),
                                    "AAAAAABBBBBBBBBB"; ints=ints, L=8.0, ng=48, L_max=6)
            ρ = ClassicalDFT.initialize_profiles(sys)
            z = range(0, 8.0, length=48)
            ρ[:, 1] .+= 0.05 .* cos.(2π .* z ./ 8.0)
            ρ[:, 2] .-= 0.05 .* cos.(2π .* z ./ 8.0)
            # ν = 0.3: the damped Maier-Saupe update converges slowly for larger ν here.
            ClassicalDFT.converge!(sys, ρ; verbose=false, tol=1e-8, maxit=2000)
            @test !any(isnan, ρ)
            @test maximum(ρ[:, 1]) - minimum(ρ[:, 1]) > 0.3
            @test maximum(abs.(sum(ρ, dims=2) .- 1.0)) < 0.05
        end
    end

    @testset "branched: all-Gaussian star through the WLC tree sweep = Gaussian tree sweep" begin
        chain = Dict("A" => GaussianChain(b=1.0), "B" => GaussianChain(b=1.3))
        seq = "AA(BB)(BBB)A"
        gl = [("mol", ["A"=>3, "B"=>5])]
        sys_g = _composite_system(chain, seq; grouplist=gl)
        sys_w = _composite_system(chain, seq; grouplist=gl,
            propagator = s -> ClassicalDFT.WLCPropagator(s.model, s.species, s.structure, CPU(), Float64; L_max=4))
        w = 0.4 .* randn(rng, 24, 2)
        a, b = _propagate(sys_g, w), _propagate(sys_w, w)
        @test b.Q ≈ a.Q rtol = 1e-12
        @test b.ρ ≈ a.ρ rtol = 1e-12
    end

    @testset "branched: uniform field and per-node identity ($seq, $label)" for (label, chain) in (
            "rod star"   => Dict("A" => WormLikeChain(b=1.0, lp=4.0), "B" => RigidRod(b=1.0)),
            "rod-coil"   => Dict("A" => WormLikeChain(b=1.0, lp=4.0), "B" => GaussianChain(b=1.0))),
            seq in ("AA(BB)(BBB)A", "A(B)(B)(B)")
        letters = unique(filter(isletter, seq))
        gl = [("mol", [string(l) => count(==(l), seq) for l in letters])]
        sys = _composite_system(chain, seq; grouplist=gl)
        r0 = _propagate(sys, zeros(24, 2))
        @test r0.Q[1] ≈ 1.0 rtol = 1e-10
        @test r0.ρ ≈ repeat(ClassicalDFT.compute_bulk_densities(sys)', 24, 1) rtol = 1e-10
        w = 0.5 .* randn(rng, 24, 2)
        r = _propagate(sys, w)
        qw = sys.propagator.sht.quad_weight
        dz = ClassicalDFT.structure_dz(sys.structure)[1]
        seqv = sys.species.sequence[1]
        for k in eachindex(seqv)
            ef = exp.(r.w_bulk[seqv[k]] .- w[:, seqv[k]])
            qq = selectdim(r.q_in[1], 3, k) .* selectdim(r.q_out[1], 3, k)
            @test sum(ClassicalDFT._orientation_marginalize(qq, qw, 2) ./ ef) * dz / (24 * dz) ≈ r.Q[1] rtol = 1e-10
        end
    end
end
