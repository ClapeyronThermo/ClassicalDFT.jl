using Test, ClassicalDFT, FFTW

# Deterministic stand-in for `randn` (avoids adding a `Random` test dependency for what
# only needs "some generic, non-special values" to exercise a transform/round-trip).
_detvals(n::Int) = [sin(1.7*i + 0.3) * cos(0.9*i) for i in 1:n]

# Unit tests for the spherical-harmonic transform (SHT) machinery and the discrete
# worm-like-chain (WLC) propagator, kept deliberately independent of the full SCFT
# machinery where possible so failures localize to the propagator math itself rather
# than the SCFT self-consistency loop (see `test_scft.jl` for the full-pipeline WLC
# tests, e.g. the uniform-melt fixed point).

@testset verbose = true "WLC" begin

@testset "SHTPlan quadrature/transform correctness" begin
    L_max = 6
    sht = ClassicalDFT.SHTPlan(L_max)
    n_orient = sht.n_theta * sht.n_phi

    @test sht.n_theta == L_max + 1
    @test sht.n_phi == 2*L_max + 1
    @test sht.nlm == (L_max+1)^2
    @test isapprox(sum(sht.quad_weight), 4*pi; rtol=1e-12)

    # constant function -> only the (l=0,m=0) coefficient is nonzero, = sqrt(4pi)
    q_grid = ones(Float64, n_orient)
    qlm = zeros(Float64, sht.nlm)
    ClassicalDFT.sht_forward!(qlm, q_grid, sht)
    @test isapprox(qlm[1], sqrt(4*pi); rtol=1e-10)
    @test maximum(abs.(qlm[2:end])) < 1e-10

    # exact round trip: coef -> grid -> coef
    qlm0 = _detvals(sht.nlm)
    grid = zeros(Float64, n_orient)
    ClassicalDFT.sht_inverse!(grid, qlm0, sht)
    qlm1 = zeros(Float64, sht.nlm)
    ClassicalDFT.sht_forward!(qlm1, grid, sht)
    @test maximum(abs.(qlm1 .- qlm0)) < 1e-9

    # grid -> coef -> grid is a genuine (idempotent) PROJECTION for a random grid
    # function, not the identity: for order m>0 there are n_theta grid values but only
    # L_max-m+1 Legendre coefficients (P_l^m undefined for l<m), so a random grid
    # function generally has energy outside the degree<=L_max representable space that
    # is correctly discarded. The correct invariant is idempotency of that projection.
    grid0 = _detvals(n_orient)
    qlm2 = zeros(Float64, sht.nlm)
    ClassicalDFT.sht_forward!(qlm2, grid0, sht)
    grid1 = zeros(Float64, n_orient)
    ClassicalDFT.sht_inverse!(grid1, qlm2, sht)
    qlm3 = zeros(Float64, sht.nlm)
    ClassicalDFT.sht_forward!(qlm3, grid1, sht)
    @test maximum(abs.(qlm3 .- qlm2)) < 1e-9

    # batched (leading spatial axes) transform matches the per-point loop
    nx, ny = 5, 3
    q_batched = reshape(_detvals(nx * ny * n_orient), nx, ny, n_orient)
    qlm_batched = zeros(Float64, nx, ny, sht.nlm)
    ClassicalDFT.sht_forward!(qlm_batched, q_batched, sht)
    maxerr = 0.0
    for i in 1:nx, j in 1:ny
        qlm_ref = zeros(Float64, sht.nlm)
        ClassicalDFT.sht_forward!(qlm_ref, q_batched[i, j, :], sht)
        maxerr = max(maxerr, maximum(abs.(qlm_ref .- qlm_batched[i, j, :])))
    end
    @test maxerr < 1e-9
end

@testset "bending_eigenvalues" begin
    L_max = 6
    κ0 = ClassicalDFT.bending_eigenvalues(0.0, L_max)
    @test κ0[1] == 1.0
    @test all(κ0[2:end] .== 0.0)

    for κ in (0.01, 1.0, 10.0, 50.0, 1000.0, 1e6)  # 1e6 exercises the besselix overflow fix
        κ̂ = ClassicalDFT.bending_eigenvalues(κ, L_max)
        @test κ̂[1] == 1.0
        @test all(isfinite, κ̂)
        @test issorted(κ̂; rev=true)  # monotonically decreasing in l for kappa>0
    end

    # small-kappa Langevin expansion: kappa_hat_1 ~ kappa/3
    κ̂_small = ClassicalDFT.bending_eigenvalues(1e-3, L_max)
    @test isapprox(κ̂_small[2], 1e-3/3; rtol=1e-2)

    # large-kappa limit: kappa_hat_1 -> 1 - 1/kappa
    κ̂_large = ClassicalDFT.bending_eigenvalues(1000.0, L_max)
    @test isapprox(κ̂_large[2], 1 - 1/1000.0; rtol=1e-2)
end

# Exact discrete freely-rotating-chain (FRC) end-to-end mean-squared-distance formula
# for an N-bead (N-1 bond) bead-rod chain with bond-correlation c1 = kappa_hat_1(kappa).
function frc_exact(b::Float64, κ::Float64, N::Int, L_max::Int)
    n = N - 1
    c1 = ClassicalDFT.bending_eigenvalues(κ, L_max)[2]
    isapprox(c1, 1.0; atol=1e-14) && return n^2 * b^2
    return n*b^2*(1+c1)/(1-c1) - 2*b^2*c1*(1-c1^n)/(1-c1)^2
end

@testset "bending operator vs exact FRC formula (grid-free moment recursion)" begin
    # Propagates the moments (rho0, rho1, rho2) of q(r,u,s) as functions of orientation
    # u only through repeated bend-then-translate steps -- entirely avoiding any
    # real-space grid (so no delta-function/discretization-resolution concerns) -- and
    # checks the resulting <R^2> against the exact closed-form FRC formula above. This
    # isolates correctness of `bending_eigenvalues`/`wlc_bend!`'s core math against the
    # derivation that "bend-then-translate" is the correct step order (see
    # `WLCPropagator`'s docstring). L_max must be >= N here since bu=b*u is itself a
    # degree-1 function of u, so the true angular degree of rho1/rho2 grows by ~1 per
    # step -- this is a property of tracking growing-degree MOMENTS exactly, not a
    # general requirement on L_max for density propagation.
    function frc_moments(b::Float64, κ::Float64, N::Int, L_max::Int)
        sht = ClassicalDFT.SHTPlan(L_max)
        eig_lm = ClassicalDFT.expand_eig_to_lm(ClassicalDFT.bending_eigenvalues(κ, L_max), sht)
        n_orient = sht.n_theta * sht.n_phi
        u = sht.u_nodes
        qw = sht.quad_weight

        function bend(f::AbstractVector{Float64})
            qlm = zeros(Float64, sht.nlm)
            ClassicalDFT.sht_forward!(qlm, f, sht)
            qlm .*= eig_lm
            g = zeros(Float64, n_orient)
            ClassicalDFT.sht_inverse!(g, qlm, sht)
            return g
        end

        rho0 = ones(Float64, n_orient)
        rho1 = zeros(Float64, 3, n_orient)
        rho2 = zeros(Float64, n_orient)

        for k in 2:N
            g0 = bend(rho0)
            g1 = vcat((bend(rho1[d, :])' for d in 1:3)...)
            g2 = bend(rho2)
            bu = b .* u
            new_rho0 = g0
            new_rho1 = g1 .+ bu .* reshape(g0, 1, n_orient)
            dot_bu_g1 = vec(sum(bu .* g1; dims=1))
            new_rho2 = g2 .+ 2 .* dot_bu_g1 .+ (b^2) .* g0
            rho0, rho1, rho2 = new_rho0, new_rho1, new_rho2
        end

        total = sum(rho0 .* qw)
        return sum(rho2 .* qw) / total, total
    end

    for (b, κ, N) in [(1.0, 0.0, 10), (1.0, 0.5, 10), (1.0, 2.0, 10), (1.0, 10.0, 12), (0.7, 5.0, 8)]
        L_max = N + 4
        R2_num, total = frc_moments(b, κ, N, L_max)
        R2_ana = frc_exact(b, κ, N, L_max)
        @test isapprox(total, 4*pi; atol=1e-8)
        @test isapprox(R2_num, R2_ana; rtol=1e-8)
    end

    # flexible limit (kappa->0) -> ideal random walk (N-1 bonds) * b^2
    R2_flex, _ = frc_moments(1.0, 0.0, 20, 24)
    @test isapprox(R2_flex, 19.0; rtol=1e-8)

    # stiff limit -> fully extended rod (N-1)^2 b^2
    R2_stiff, _ = frc_moments(1.0, 1e4, 15, 19)
    @test isapprox(R2_stiff, (15-1)^2*1.0; rtol=1e-3)
end

@testset "grid-based (FFT+SHT) propagator vs exact FRC formula" begin
    # End-to-end validation of the actual production propagator code
    # (wlc_translate!/wlc_bend!, the FFT+SHT path) against the same FRC target,
    # complementing the exact grid-free test above by exercising the real grid/FFT
    # wiring: a delta-function point source at the domain center, propagated for N-1
    # bend-then-translate steps in zero field, with the orientation-marginalized second
    # moment about the source compared to the analytic formula.
    function run_chain(b, κ, N, L_max, nx)
        nd = 3
        ngrid = (nx, nx, nx)
        rfft_ngrid = (nx÷2+1, nx, nx)
        ν1 = FFTW.rfftfreq(nx, 1.0)
        ν2 = FFTW.fftfreq(nx, 1.0)
        ν3 = FFTW.fftfreq(nx, 1.0)
        ν = (reshape(ν1, rfft_ngrid[1],1,1), reshape(ν2,1,nx,1), reshape(ν3,1,1,nx))

        sht = ClassicalDFT.SHTPlan(L_max)
        n_orient = sht.n_theta * sht.n_phi

        CT = ComplexF64
        phase = zeros(CT, rfft_ngrid..., n_orient)
        for i in 1:n_orient
            d = b .* @view sht.u_nodes[:, i]
            arg = zeros(Float64, rfft_ngrid...)
            for dim in 1:nd
                arg = arg .+ ν[dim] .* d[dim]
            end
            selectdim(phase, nd+1, i) .= CT.(exp.(2*pi*im .* arg))
        end
        bend_eig = ClassicalDFT.expand_eig_to_lm(ClassicalDFT.bending_eigenvalues(κ, L_max), sht)

        buf_r = zeros(Float64, ngrid...)
        buf_c = zeros(ComplexF64, rfft_ngrid...)
        qlm_buf = zeros(Float64, ngrid..., sht.nlm)
        P = FFTW.plan_rfft(buf_r, 1:nd)
        iP = FFTW.plan_irfft(buf_c, ngrid[1], 1:nd)

        c0 = nx ÷ 2 + 1
        q1 = zeros(Float64, ngrid..., n_orient)
        q1[c0, c0, c0, :] .= 1.0
        qcur = q1
        for k in 2:N
            bend_buf = similar(qcur)
            ClassicalDFT.wlc_bend!(bend_buf, qcur, bend_eig, sht, qlm_buf, nd)
            trans_buf = similar(qcur)
            ClassicalDFT.wlc_translate!(trans_buf, bend_buf, phase, buf_r, buf_c, P, iP, nd)
            qcur = trans_buf
        end

        rho = ClassicalDFT._orientation_marginalize(qcur, sht.quad_weight, nd+1)
        total = sum(rho)

        coord(i) = (i - c0) <= nx÷2 ? Float64(i - c0) : Float64(i - c0 - nx)
        R2 = 0.0
        for i in 1:nx, j in 1:nx, k in 1:nx
            R2 += (coord(i)^2 + coord(j)^2 + coord(k)^2) * rho[i,j,k]
        end
        return R2 / total
    end

    for (b, κ, N, nx) in [(1.0, 0.0, 8, 48), (1.0, 1.0, 8, 48), (1.0, 3.0, 8, 48)]
        L_max = 14
        R2_num = run_chain(b, κ, N, L_max, nx)
        R2_ana = frc_exact(b, κ, N, L_max)
        @test isapprox(R2_num, R2_ana; rtol=0.05)
    end
end

@testset "1D/2D translationally-invariant reduction" begin
    # WLCPropagator supports dimension(structure) in {1,2,3} (orientation is always
    # tracked in full 3D via SHT; only the translation step's kernel uses the first `nd`
    # Cartesian components of u). By isotropy, the nd-dimensional tracked-position
    # mean-squared displacement should equal exactly (nd/3) of the full 3D <R^2>.
    function run_chain_nd(b, κ, N, L_max, nx, nd)
        ngrid = ntuple(_ -> nx, nd)
        rfft_ngrid = (nx ÷ 2 + 1, ngrid[2:end]...)
        νall = ntuple(nd) do i
            v = i == 1 ? FFTW.rfftfreq(nx, 1.0) : FFTW.fftfreq(nx, 1.0)
            reshape(v, ntuple(d -> d == i ? rfft_ngrid[d] : 1, nd))
        end

        sht = ClassicalDFT.SHTPlan(L_max)
        n_orient = sht.n_theta * sht.n_phi
        CT = ComplexF64
        phase = zeros(CT, rfft_ngrid..., n_orient)
        for i in 1:n_orient
            d = b .* @view sht.u_nodes[:, i]
            arg = zeros(Float64, rfft_ngrid...)
            for dim in 1:nd
                arg = arg .+ νall[dim] .* d[dim]
            end
            selectdim(phase, nd+1, i) .= CT.(exp.(2*pi*im .* arg))
        end
        bend_eig = ClassicalDFT.expand_eig_to_lm(ClassicalDFT.bending_eigenvalues(κ, L_max), sht)

        buf_r = zeros(Float64, ngrid...)
        buf_c = zeros(ComplexF64, rfft_ngrid...)
        qlm_buf = zeros(Float64, ngrid..., sht.nlm)
        P = FFTW.plan_rfft(buf_r, 1:nd)
        iP = FFTW.plan_irfft(buf_c, ngrid[1], 1:nd)

        c0 = nx ÷ 2 + 1
        q1 = zeros(Float64, ngrid..., n_orient)
        idx = ntuple(_ -> c0, nd)
        q1[idx..., :] .= 1.0
        qcur = q1
        for k in 2:N
            bend_buf = similar(qcur)
            ClassicalDFT.wlc_bend!(bend_buf, qcur, bend_eig, sht, qlm_buf, nd)
            trans_buf = similar(qcur)
            ClassicalDFT.wlc_translate!(trans_buf, bend_buf, phase, buf_r, buf_c, P, iP, nd)
            qcur = trans_buf
        end

        rho = ClassicalDFT._orientation_marginalize(qcur, sht.quad_weight, nd + 1)
        total = sum(rho)

        coord(i) = (i - c0) <= nx÷2 ? Float64(i - c0) : Float64(i - c0 - nx)
        R2 = 0.0
        for ci in CartesianIndices(ngrid)
            k = Tuple(ci)
            r2 = sum(coord(k[d])^2 for d in 1:nd)
            R2 += r2 * rho[ci]
        end
        return R2 / total
    end

    b, κ, N, L_max, nx = 1.0, 2.0, 8, 14, 48
    R2_3d = frc_exact(b, κ, N, L_max)
    R2_1d = run_chain_nd(b, κ, N, L_max, nx, 1)
    @test isapprox(R2_1d, R2_3d/3; rtol=0.02)
    R2_2d = run_chain_nd(b, κ, N, L_max, nx, 2)
    @test isapprox(R2_2d, 2*R2_3d/3; rtol=0.02)
end

@testset "SCFTWormLikeChainFluid: L_max is user-configurable" begin
    model = ClassicalDFT.SCFTWormLikeChainFluid([("rod", ["A"=>4])], [1.0], [3.0], zeros(1,1);
                                                 rho0=1.0, kappa=20.0, L_max=5)
    mol_structure = Dict("rod" => ClassicalDFT.custom_structure("A"^4))
    structure = ClassicalDFT.Uniform1DCart((0.0, 0.0), [1.0], [0.0, 16.0], 32)
    system = ClassicalDFT.SCFTSystem(model, structure, ClassicalDFT.DFTOptions();
        mol_structure=mol_structure, ensemble=[:canonical], n_molecules=[4.0])
    @test system.propagator.sht.L_max == 5
end

end # testset "WLC"
