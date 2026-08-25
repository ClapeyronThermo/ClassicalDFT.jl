using Test, ClassicalDFT

@testset "Anderson acceleration (aasol)" begin
    # Toth-Kelley fixed-point problem from aasol's own docstring (src/utils/anderson.jl)
    # — self-contained, unlike the docstring's other example (heqinit/HeqFix!, which
    # references functions that don't exist anywhere in this repo).
    function tothk!(G, u)
        G[1] = cos(0.5*(u[1]+u[2]))
        G[2] = G[1] + 1.e-8*sin(u[1]*u[1])
        return G
    end

    @testset "Default mixing (beta=1.0)" begin
        u0 = ones(2); m = 2; Vstore = zeros(2, 3*m+3)
        aout = ClassicalDFT.aasol(tothk!, u0, m, Vstore; rtol=1e-10)
        @test aout.idid
        @test aout.errcode == 0
        @test aout.history[end] < 1e-10
        @test aout.solution ≈ aout.functionval atol=1e-8  # fixed point: G(x*) ≈ x*
    end

    @testset "beta=0.5 mixing" begin
        u0 = ones(2); m = 2; Vstore = zeros(2, 3*m+3)
        bout = ClassicalDFT.aasol(tothk!, u0, m, Vstore; rtol=1e-10, beta=0.5)
        @test bout.idid
        @test bout.errcode == 0
        @test bout.history[end] < 1e-10
    end

    @testset "Picard warmup" begin
        u0 = ones(2); m = 2; Vstore = zeros(2, 3*m+3)
        aout = ClassicalDFT.aasol(tothk!, u0, m, Vstore; picard_maxit=5, picard_beta=0.5, rtol=1e-10)
        @test aout.idid
        @test aout.errcode == 0
        @test aout.history[end] < 1e-10
    end
end
