using Test, ClassicalDFT
using ClassicalDFT.Clapeyron

@testset "DGT" begin

    @testset "Bulk consistency (κ=0 reduces to bulk EoS)" begin
        # A plain (non-group-contribution) model: DGT sizes its fields as nbeads=ones(nc)
        # (one "bead" per component), which breaks @chain(i) for group-contribution models.
        p, T = 1e5, 298.15
        x = [1.0]
        model = PCSAFT(["water"])
        # database/gradients/ doesn't exist in this repo, so the plain-string constructor
        # throws; the NamedTuple userlocations form bypasses it entirely (same trick
        # test_models.jl already uses for COFFEE).
        gradient = ClassicalDFT.ConstGradient(["water"]; userlocations=(; kappa=[0.0]))

        vl = volume(model, p, T, x)
        ρbulk = x/vl
        L = ClassicalDFT.length_scale(model)

        structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], 51)
        system = ClassicalDFT.DGTSystem(model, gradient, structure)

        ρ = ClassicalDFT.initialize_profiles(system)
        μ1 = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / T / Clapeyron.Rgas()
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)
        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "Gradient term scaling" begin
        # A bulk-consistency check alone can never exercise κ (the gradient term
        # vanishes identically whenever ∇ρ=0, regardless of κ — dgt.jl's f_res). Build
        # one non-uniform profile and evaluate F_res for three κ values on the exact
        # same profile: the gradient term (Σᵢ κᵢᵢ/2·|∇ρ̄ᵢ|² + ...) is linear in κ and
        # positive-semi-definite for κ>0 — both are exact algebraic properties of the
        # formula (not an iterative solve), so should hold tightly.
        p, T = 1e5, 298.15
        x = [1.0]
        model = PCSAFT(["water"])
        vl = volume(model, p, T, x)
        ρbulk = x/vl
        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], 51)

        κ0 = 1e-20
        gradient0  = ClassicalDFT.ConstGradient(["water"]; userlocations=(; kappa=[0.0]))
        gradient1  = ClassicalDFT.ConstGradient(["water"]; userlocations=(; kappa=[κ0]))
        gradient2  = ClassicalDFT.ConstGradient(["water"]; userlocations=(; kappa=[2κ0]))
        system0 = ClassicalDFT.DGTSystem(model, gradient0, structure)
        system1 = ClassicalDFT.DGTSystem(model, gradient1, structure)
        system2 = ClassicalDFT.DGTSystem(model, gradient2, structure)

        ρ = ClassicalDFT.initialize_profiles(system0; noise=0.05)

        F0 = ClassicalDFT.F_res(system0, ρ)
        F1 = ClassicalDFT.F_res(system1, ρ)
        F2 = ClassicalDFT.F_res(system2, ρ)

        @test F1 > F0
        @test (F2 - F0) ≈ 2*(F1 - F0) rtol = 1e-6
    end
end
