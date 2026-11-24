using Test, ClassicalDFT

# Bulk (uniform) thermodynamics with nematic order: src/models/SCFT/nematic.jl.
const _N = ClassicalDFT

# Independent reference for a compressible rigid-rod melt (N beads, Helfand κ, Maier-Saupe
# ν): the same mean-field free energy evaluated with Simpson quadrature over cosθ.
module RodMeltReference
    const xs = range(-1, 1, length=20001)
    const sw = (h = step(xs); w = fill(2h / 3, length(xs)); w[2:2:end-1] .= 4h / 3; w[1] = w[end] = h / 3; w ./ 2)
    P2(x) = (3x^2 - 1) / 2
    logZ(a) = (e = a .* P2.(xs); m = maximum(e); m + log(sum(sw .* exp.(e .- m))))
    function S_self(a; S=0.8)          # S = ⟨P₂⟩ under exp(a S P₂), a = total coupling per rod
        for _ in 1:100000
            e = a * S .* P2.(xs); w = sw .* exp.(e .- maximum(e))
            Sn = sum(P2.(xs) .* w) / sum(w)
            abs(Sn - S) < 1e-14 && return Sn
            S = Sn
        end
        return S
    end
    f(ρm, S; N, κ, ν, ρ0) = (ρs = N * ρm; ρm * (log(ρm) - 1) + κ / (2ρ0^2) * (ρs - ρ0)^2 - ν * ρs^2 / (4ρ0) +
                             ν * ρs^2 * S^2 / (2ρ0) - ρm * logZ(N * ν * ρs * S / ρ0))
    μ(ρm, S; p...) = (h = 1e-7; (f(ρm + h, S; p...) - f(ρm - h, S; p...)) / 2h)
    P(ρm, S; p...) = ρm * μ(ρm, S; p...) - f(ρm, S; p...)
    function coexistence(ρI, ρN; N, κ, ν, ρ0)
        p = (; N, κ, ν, ρ0)
        r(y) = (SN = S_self(N * ν * N * exp(y[2]) / ρ0);
                [μ(exp(y[1]), 0.0; p...) - μ(exp(y[2]), SN; p...), P(exp(y[1]), 0.0; p...) - P(exp(y[2]), SN; p...)])
        y = [log(ρI), log(ρN)]
        for _ in 1:60
            r0 = r(y); maximum(abs, r0) < 1e-11 && break
            J = hcat([(r(y .+ h) .- r(y .- h)) ./ 2e-6 for h in ([1e-6, 0], [0, 1e-6])]...)
            y .-= J \ r0
        end
        return exp.(y), S_self(N * ν * N * exp(y[2]) / ρ0)
    end
end

function _rod_model(; N=6, ν, κ=20.0, L_max=12)
    m = SCFTModel([("rod", ["A"=>N])]; rho0=1.0, L_max=L_max, chain=RigidRod(b=1.0),
                  interactions=(HelfandCompressibility(κ), MaierSaupe(fill(ν, 1, 1))))
    return m, NematicBulk(m; mol_structure=Dict("rod" => _N.custom_structure("A"^N)))
end

# Uniform SCFT solution seeded nematic; returns (H/V, segment densities, S per species, system).
function _uniform_scft(gl, seq, chain, ints; L=6.0, ng=8, L_max=8, director=(0, 0, 1))
    m = SCFTModel(gl; rho0=1.0, L_max=L_max, chain=chain, interactions=ints)
    st = _N.Uniform1DCart((0.0, 0.0), [1.0], [0.0, L], ng)
    sys = _N.SCFTSystem(m, st, _N.DFTOptions(); mol_structure=Dict(gl[1][1] => _N.custom_structure(seq)),
                        ensemble=[:canonical], n_molecules=[L / count(isletter, seq)])
    ρ = _N.initialize_profiles(sys)
    r = _N.converge!(sys, ρ; tol=1e-11, maxit=3000, director=director, nematic_order=0.6)
    @test r.converged
    H = _N.free_energy(sys, ρ, r.w, r.Q; R_tensor=r.R_tensor)
    S = [(3 * sum(r.R_tensor[:, α, 3]) / sum(ρ[:, α]) - 1) / 2 for α in axes(ρ, 2)]
    return H / L, vec(sum(ρ, dims=1)) ./ ng, S, sys
end

@testset "SCFT nematic bulk" begin
    @testset "isotropic Maier-Saupe term: uniform SCFT free energy = bulk free energy" begin
        nu = [0.4 0.2; 0.2 0.3]
        HV, ρseg, S, _ = _uniform_scft([("rc", ["A"=>3, "B"=>3])], "AAABBB", WormLikeChain(b=1.0, lp=2.0),
            (FloryHugginsChi([0.0 0.5; 0.5 0.0]), HelfandCompressibility(20.0), MaierSaupe(nu)); director=nothing)
        @test all(abs.(S) .< 1e-10)
        @test HV ≈ bulk_free_energy_density(SCFTModel([("rc", ["A"=>3, "B"=>3])]; rho0=1.0,
            chain=WormLikeChain(b=1.0, lp=2.0),
            interactions=(FloryHugginsChi([0.0 0.5; 0.5 0.0]), HelfandCompressibility(20.0), MaierSaupe(nu))), ρseg) rtol = 1e-10
    end

    @testset "rigid rod: order parameter and free energy vs classical Maier-Saupe (ν = $ν)" for ν in (1.0, 2.0)
        N = 6
        _, nb = _rod_model(; N, ν, L_max=12)
        r = bulk_order_parameter(nb, [1.0 / N])
        @test r.converged
        a = N * ν
        S_ref = RodMeltReference.S_self(a)
        @test r.S[1] ≈ S_ref atol = 1e-3                 # L_max = 12 truncation
        @test r.Δf * N ≈ a * S_ref^2 / 2 - RodMeltReference.logZ(a * S_ref) atol = 5e-3
        # below the transition (a = 3 < 4.49) only the isotropic solution exists
        _, nb_lo = _rod_model(; N, ν=0.5)
        r_lo = bulk_order_parameter(nb_lo, [1.0 / N])
        @test r_lo.converged && abs(r_lo.S[1]) < 1e-8 && abs(r_lo.Δf) < 1e-10
    end

    @testset "bulk free energy = uniform nematic SCFT free energy ($label)" for (label, gl, seq, chain, ints) in (
            ("rod melt", [("rod", ["A"=>6])], "AAAAAA", RigidRod(b=1.0),
             (HelfandCompressibility(20.0), MaierSaupe(fill(1.0, 1, 1)))),
            ("rod-coil", [("rc", ["A"=>4, "B"=>4])], "AAAABBBB",
             Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0)),
             (FloryHugginsChi(Dict(("A","B") => 0.3)), HelfandCompressibility(20.0), MaierSaupe(Dict(("A","A") => 4.0)))),
            ("branched WLC/Gaussian", [("st", ["A"=>3, "B"=>5])], "AA(BB)(BBB)A",
             Dict("A" => WormLikeChain(b=1.0, lp=6.0), "B" => GaussianChain(b=1.0)),
             (HelfandCompressibility(20.0), MaierSaupe(Dict(("A","A") => 25.0)))))
        HV, ρseg, S_scft, sys = _uniform_scft(gl, seq, chain, ints)
        nb = NematicBulk(sys)
        b = bulk_order_parameter(nb, [1.0 / count(isletter, seq)])
        @test S_scft[1] > 0.5                              # genuinely nematic
        @test b.S ≈ S_scft atol = 1e-7
        @test HV ≈ bulk_free_energy_density(sys.model, ρseg) + b.Δf rtol = 1e-8
    end

    @testset "rod melt I–N coexistence vs independent calculation" begin
        N, κ, ν = 6, 5.0, 0.76
        _, nb = _rod_model(; N, ν, κ, L_max=12)
        r = isotropic_nematic_coexistence(nb, [0.98 / N], [1.02 / N])
        @test r.converged
        (ρI, ρN), S_ref = RodMeltReference.coexistence(0.98 / N, 1.02 / N; N, κ, ν, ρ0=1.0)
        @test r.ρ_iso[1] ≈ ρI rtol = 1e-8
        @test r.ρ_nem[1] ≈ ρN rtol = 1e-8
        @test r.S_nem[1] ≈ S_ref atol = 1e-6
        @test r.ρ_nem[1] > r.ρ_iso[1]                      # nematic is the denser phase
    end

    @testset "rods + Gaussian solvent: coexistence at fixed isotropic composition" begin
        m = SCFTModel([("rod", ["A"=>6]), ("solv", ["S"=>1])]; rho0=1.0, L_max=12,
                      chain=Dict("A" => RigidRod(b=1.0), "S" => GaussianChain(b=1.0)),
                      interactions=(HelfandCompressibility(10.0), MaierSaupe(Dict(("A","A") => 1.2))))
        nb = NematicBulk(m; mol_structure=Dict("rod" => _N.custom_structure("AAAAAA"), "solv" => _N.custom_structure("S")))
        φI = 0.65
        r = isotropic_nematic_coexistence(nb, [φI / 6, 1 - φI], [0.8 / 6, 0.2])
        @test r.converged
        S0 = zeros(2)
        @test bulk_chemical_potential(nb, r.ρ_iso, S0) ≈ bulk_chemical_potential(nb, r.ρ_nem, r.S_nem) atol = 1e-8
        @test bulk_pressure(nb, r.ρ_iso, S0) ≈ bulk_pressure(nb, r.ρ_nem, r.S_nem) atol = 1e-8
        @test 6r.ρ_iso[1] / (6r.ρ_iso[1] + r.ρ_iso[2]) ≈ φI rtol = 1e-12   # composition held fixed
        @test 6r.ρ_nem[1] / (6r.ρ_nem[1] + r.ρ_nem[2]) > φI                # nematic enriched in rods
        @test r.S_nem[2] == 0                                               # Gaussian solvent
    end

    @testset "no orientation-resolved species: always isotropic" begin
        m = SCFTModel([("x", ["A"=>3])]; rho0=1.0, interactions=(HelfandCompressibility(10.0),))
        nb = NematicBulk(m; mol_structure=Dict("x" => _N.custom_structure("AAA")))
        r = bulk_order_parameter(nb, [1 / 3])
        @test r.converged && r.S == [0.0] && abs(r.Δf) < 1e-12
    end
end
