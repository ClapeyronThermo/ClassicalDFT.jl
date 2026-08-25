function bulk_μres_clapeyron(model, p, T, ρbulk)
    x = ρbulk ./ sum(ρbulk)
    v = 1 / sum(ρbulk)
    return Clapeyron.VT_chemical_potential_res(model, v, T, x) ./ Clapeyron.R̄ ./ T
end

# Evaluate δF_res/δρ at uniform bulk density and compare to Clapeyron μ_res.
function test_bulk_μres(label, system, ρbulk; tol=1e-5)
    ρ = ClassicalDFT.initialize_profiles(system)
    μ_dft = ClassicalDFT.δFδρ_res(system, ρ)
    nd = ClassicalDFT.dimension(system)
    nb = sum(system.species.nbeads)
    nc = length(system.model)

    idx = ntuple(_ -> 1, nd)
    μ_dft_comp = zeros(nc)
    k = 1
    for i in 1:nc
        for _ in 1:system.species.nbeads[i]
            μ_dft_comp[i] += μ_dft[idx..., k] / system.species.nbeads[i]
            k += 1
        end
    end

    μ_ref = bulk_μres_clapeyron(system.model, system.structure.conditions..., ρbulk)

    @testset "$label" begin
        for i in 1:nc
            err = abs(μ_dft_comp[i] - μ_ref[i])
            @test err < tol
        end
    end
end

function test_bulk_μres_electrolyte(label, system, ρbulk; tol=1e-4)
    ρ     = ClassicalDFT.initialize_profiles(system)
    μ_dft = ClassicalDFT.δFδρ_res(system, ρ)
    nbeads = system.species.nbeads
    nc     = length(nbeads)

    μ_dft_comp = zeros(nc)
    k = 1
    for i in 1:nc
        for _ in 1:nbeads[i]
            μ_dft_comp[i] += μ_dft[1, k] / nbeads[i]
            k += 1
        end
    end

    μ_ref = bulk_μres_clapeyron(system.model, system.structure.conditions..., ρbulk)

    @testset "$label" begin
        for i in 1:nc
            err = abs(μ_dft_comp[i] - μ_ref[i])
            @test err < tol
        end
    end
end

@testset "Coordinate systems: bulk μ_res consistency" begin

    model = PCSAFT(["methane"])
    T = 150.0
    p = 1e7
    v = Clapeyron.volume(model, p, T, [1.0]; phase=:liquid)
    ρbulk = [1/v]
    L = ClassicalDFT.length_scale(model)

    @testset "PCSAFT methane" begin
        s_cart = Uniform1DCart((p,T), ρbulk, [-10L, 10L], 101)
        test_bulk_μres("1D Cart", DFTSystem(model, s_cart), ρbulk)

        # Spherical (n=2) QDHT converges slower than cylindrical (n=1) at fixed N;
        # N=151 keeps this comfortably under the 1e-6 tolerance (N=101 does not - see
        # coordinate_system.md implementation notes).
        s_sphr = Uniform1DSphr((p,T), ρbulk, [0.0, 10L], 151)
        test_bulk_μres("1D Sphr", DFTSystem(model, s_sphr), ρbulk)

        s_cyl = Uniform1DCyl((p,T), ρbulk, [0.0, 20L], 101)
        test_bulk_μres("1D Cyl", DFTSystem(model, s_cyl), ρbulk)
    end

    @testset "HeterogcPCPSAFT 1-butanol (TangentHSPropagator)" begin
        model_but = HeterogcPCPSAFT(["1-butanol"])
        T_but = 298.15
        (p_but, vl_but, _) = Clapeyron.saturation_pressure(model_but, T_but)
        ρbulk_but = [1/vl_but]
        L_but = ClassicalDFT.length_scale(model_but)

        # The propagator's bond-length kernel (~sum of two group radii, scaled by π) is
        # much larger than the FMT weighted-density half-diameters, so the QDHT aperture
        # needs a substantially larger multiple of L than a monomer-only FMT system to
        # resolve it well (60L here; 10L is comparable to the bond length itself and
        # gives badly wrong results — this is a genuine QDHT resolution requirement, not
        # a solver issue).
        s_sphr = Uniform1DSphr((p_but,T_but), ρbulk_but, [0.0, 60L_but], 151)
        test_bulk_μres("1D Sphr (propagator)", DFTSystem(model_but, s_sphr), ρbulk_but)

        s_cyl = Uniform1DCyl((p_but,T_but), ρbulk_but, [0.0, 60L_but], 151)
        test_bulk_μres("1D Cyl (propagator)", DFTSystem(model_but, s_cyl), ρbulk_but)
    end

    @testset "ePCSAFT electrolyte (ElectrostaticPotential)" begin
        model_elec = ePCSAFT(["water08"], ["sodium", "chloride"])
        model_elec.neutralmodel.params.epsilon.values[2,2] = 197.737^2/70.0
        model_elec.neutralmodel.params.epsilon.values[3,3] = 70.0
        T_elec = 298.15
        p_elec = 1e7
        x_elec = [0.9, 0.05, 0.05]
        v_elec = Clapeyron.volume(model_elec.neutralmodel, p_elec, T_elec, x_elec)
        ρbulk_elec = x_elec / v_elec
        L_elec = ClassicalDFT.length_scale(model_elec.neutralmodel)

        # The Coulomb kernel diverges at small k (long-ranged in real space), so unlike
        # the neutral case, accuracy here is governed by the QDHT aperture (domain size)
        # more than by N — 30L (Sphr) / 20L (Cyl) at N=151 hits the sweet spot; larger
        # apertures without more points under-resolve the short-range ion-size features
        # and can fail outright (see coordinate_system.md implementation notes).
        s_sphr = Uniform1DSphr((p_elec,T_elec), ρbulk_elec, [0.0, 30L_elec], 151)
        sys_sphr = ClassicalDFT.ElectrolyteDFTSystem(model_elec, s_sphr)
        test_bulk_μres_electrolyte("1D Sphr (electrolyte)", sys_sphr, ρbulk_elec)

        s_cyl = Uniform1DCyl((p_elec,T_elec), ρbulk_elec, [0.0, 20L_elec], 151)
        sys_cyl = ClassicalDFT.ElectrolyteDFTSystem(model_elec, s_cyl)
        test_bulk_μres_electrolyte("1D Cyl (electrolyte)", sys_cyl, ρbulk_elec)
    end
end

@testset "Coordinate systems: non-Cartesian kernel smoke tests" begin
    model_pc = PCSAFT(["methane"])
    T_pc = 150.0; p_pc = 1e7
    v_pc = Clapeyron.volume(model_pc, p_pc, T_pc, [1.0]; phase=:liquid)
    ρbulk_pc = [1/v_pc]
    L_pc = ClassicalDFT.length_scale(model_pc)

    @testset "PCSAFT methane — non-Cartesian δFδρ_res" begin
        @testset "1D Sphr" begin
            s = Uniform1DSphr((p_pc, T_pc), ρbulk_pc, [0.0, 10L_pc], 101)
            sys = DFTSystem(model_pc, s)
            ρ = ClassicalDFT.initialize_profiles(sys)
            @test_nowarn ClassicalDFT.δFδρ_res(sys, ρ)
        end
        @testset "1D Cyl" begin
            s = Uniform1DCyl((p_pc, T_pc), ρbulk_pc, [0.0, 20L_pc], 101)
            sys = DFTSystem(model_pc, s)
            ρ = ClassicalDFT.initialize_profiles(sys)
            @test_nowarn ClassicalDFT.δFδρ_res(sys, ρ)
        end
    end
end