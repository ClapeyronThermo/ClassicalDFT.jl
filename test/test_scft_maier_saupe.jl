using Test, ClassicalDFT

# Maier-Saupe solver: R is part of the Anderson unknown (see converge!'s docstring).
const _C = ClassicalDFT

function _wlc_rodcoil(; nu, chi)
    m = SCFTWormLikeChainFluid([("rc", ["A"=>6, "B"=>6])], [1.0, 1.0], [8.0, 1.0], [0.0 chi; chi 0.0];
                               rho0=1.0, kappa=20.0, nu=[nu 0.0; 0.0 0.0], L_max=6)
    st = _C.Uniform1DCart((0.0, 0.0), [1.0], [0.0, 12.0], 48)
    sys = _C.SCFTSystem(m, st, _C.DFTOptions(); mol_structure=Dict("rc" => _C.custom_structure("A"^6 * "B"^6)),
                        ensemble=[:canonical], n_molecules=[1.0])
    ρ = _C.initialize_profiles(sys)
    z = range(0, 12.0, length=48)
    ρ[:, 1] .+= 0.05 .* cos.(2π .* z ./ 12.0); ρ[:, 2] .-= 0.05 .* cos.(2π .* z ./ 12.0)
    return sys, ρ
end

function _rod_melt(nu; L_max=6)
    m = SCFTModel([("rod", ["A"=>6])]; rho0=1.0, L_max=L_max, chain=RigidRod(b=1.0),
                  interactions=(HelfandCompressibility(20.0), MaierSaupe(fill(nu, 1, 1))))
    st = _C.Uniform1DCart((0.0, 0.0), [1.0], [0.0, 6.0], 16)
    sys = _C.SCFTSystem(m, st, _C.DFTOptions(); mol_structure=Dict("rod" => _C.custom_structure("AAAAAA")),
                        ensemble=[:canonical], n_molecules=[1.0])
    return sys, _C.initialize_profiles(sys)
end

# Uniaxial order along z from R = ρ⟨uu⟩: S = (3⟨u_z²⟩ − 1)/2 (pairs: 1=(1,1), 3=(3,3)).
_Sz(R, ρ) = (3 * (sum(R[:, 1, 3]) / sum(ρ)) - 1) / 2

@testset "SCFT Maier-Saupe" begin
    @testset "converges where the damped update stalled or hit NaN (ν = 2, χ = $chi)" for chi in (1.2, 2.0)
        sys, ρ = _wlc_rodcoil(nu=2.0, chi=chi)
        r = _C.converge!(sys, ρ; tol=1e-8, maxit=4000)
        @test r.converged
        @test !any(isnan, ρ)
        @test maximum(ρ[:, 1]) - minimum(ρ[:, 1]) > 0.5
        @test maximum(abs.(sum(ρ, dims=2) .- 1)) < 0.1     # soft Helfand κ = 20
    end

    # A rigid rod of N = 6 beads shares one orientation, so its coupling is a = Nν:
    # classical Maier-Saupe gives an isotropic-nematic transition at a ≈ 4.49 (ν ≈ 0.75).
    @testset "rod melt: isotropic below the transition, nematic above it" begin
        # ν = 0.5 (a = 3): an aligned seed relaxes back to isotropic.
        sys, ρ = _rod_melt(0.5)
        r = _C.converge!(sys, ρ; tol=1e-9, maxit=2000, director=(0, 0, 1), nematic_order=0.5)
        @test r.converged
        @test abs(_Sz(r.R_tensor, ρ)) < 1e-6
        # ν = 6: an aligned seed grows into a nematic, which must now be reported (the old
        # damped scheme declared convergence after one iteration with R barely moved).
        sys, ρ = _rod_melt(6.0)
        r = _C.converge!(sys, ρ; tol=1e-9, maxit=2000, director=(0, 0, 1), nematic_order=0.1)
        @test r.converged
        @test _Sz(r.R_tensor, ρ) > 0.5
        # Without a seed the isotropic state is an exact fixed point.
        sys, ρ = _rod_melt(6.0)
        r = _C.converge!(sys, ρ; tol=1e-9, maxit=2000)
        @test r.converged
        @test abs(_Sz(r.R_tensor, ρ)) < 1e-8
    end

    @testset "rod melt order parameter matches classical Maier-Saupe theory" begin
        # S = ⟨P₂⟩ under exp(a S P₂(cosθ)), a = Nν = 6: S = 0.74735 (self-consistent solution).
        sys, ρ = _rod_melt(1.0; L_max=8)
        r = _C.converge!(sys, ρ; tol=1e-10, maxit=3000, director=(0, 0, 1), nematic_order=0.5)
        @test r.converged
        @test _Sz(r.R_tensor, ρ) ≈ 0.74735 atol = 2e-3
    end

    @testset "director seed is normalized and validated" begin
        sys, _ = _rod_melt(1.0)
        @test _C.SCFTProblem(sys; director=(0, 0, 2)).director == (0.0, 0.0, 1.0)
        @test_throws ArgumentError _C.SCFTProblem(sys; director=(0, 0, 0))
    end
end
