using Test, ClassicalDFT
using ClassicalDFT.Clapeyron

@testset "Models" begin
    p = 1e5
    T = 298.15
    x = [0.5,0.5]
    x1 = [1.]
    x3 = [0.333, 0.333,0.333]

    @testset "PCSAFT" begin
        model = PCSAFT(["hexane","acetone"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl
        
        L = ClassicalDFT.length_scale(model)
        
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6

    end

    @testset "PCPSAFT" begin
        model = PCPSAFT(["acetone","hexane","DMSO"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x3)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x3)
        ρ = x3/vl
        
        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "HeterogcPCPSAFT" begin
        # Use non-associating, non-polar alkanes: GCIdentifier supports CH3/CH2 for gcPCSAFT.
        # acetone contains >C=O which GCIdentifier cannot map → bond topology error.
        model = HeterogcPCPSAFT(["hexane","heptane"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl
        
        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "SAFTVRMie" begin
        model = SAFTVRMie(["methane","butane"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl
        
        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T),ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "SAFTgammaMie" begin
        # Use non-associating species: methane (1 bead) + butane (4 beads).
        # Association contributions are deferred; checking HS+chain+disp only.
        model = SAFTgammaMie(["methane","butane"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "SAFTgammaMie hexane+butane" begin
        # Hexane and butane share group types (CH3, CH2) — tests that f_chain uses
        # the correct non-contiguous group indices from model.groups.i_groups.
        model_hb = SAFTgammaMie(["hexane","butane"])
        μ1_hb = Clapeyron.chemical_potential_res(model_hb, p, T, x)/T/Clapeyron.Rgas()
        vl_hb = volume(model_hb, p, T, x)
        ρ_hb  = x/vl_hb
        L_hb  = ClassicalDFT.length_scale(model_hb)
        structure_hb = Uniform1DCart((p, T), ρ_hb, [-10L_hb, 10L_hb], (3,))
        system_hb    = DFTSystem(model_hb, structure_hb)
        ρ0_hb = ClassicalDFT.initialize_profiles(system_hb)
        μ2_hb = ClassicalDFT.δFδρ_res(system_hb, ρ0_hb)
        @test μ1_hb[1] ≈ μ2_hb[1] rtol = 1e-6
    end

    @testset "COFFEE" begin
        model = COFFEE(["a1"]; userlocations = (;
            Mw = [1.],
            segment = [1.],
            sigma = [3.;;],
            epsilon = [300.;;],
            lambda_r = [12;;],
            lambda_a = [6;;],
            shift = [3*0.15],
            dipole = [1.0*1.0575091914494172],))

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T),ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "QPCPSAFT" begin
        # CO2 (pure QQ) + acetone (DD + DQ + QQ) — tests polar quadrupole contributions
        model = QPCPSAFT(["CO2","acetone"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "PCSAFT associating (water)" begin
        # Single-component water: 1 H-site + 1 O-site, 1 association pair (i=j=1, a=1, b=2).
        # Tests that f_assoc embedded in kernel gives correct functional derivative at bulk.
        model = PCSAFT(["water"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-4
    end

    @testset "PCSAFT associating (water+ethanol)" begin
        # Binary mixture: water (2 sites) + ethanol (2 sites) → 4 pairs (including
        # cross-association).  Exercises the general 50-iteration fixed-point path.
        model = PCSAFT(["water","ethanol"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1,1] rtol = 1e-4
        @test μ1[2] ≈ μ2[1,2] rtol = 1e-4
    end

    @testset "PCPSAFT associating (water)" begin
        # Water with PCPSAFT: 1 assoc pair → Val{1} analytical path.
        # Exercises g_hs _assoc_delta via generic f_assoc for PCPSAFTModel.
        model = PCPSAFT(["water"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-4
    end

    @testset "SAFTVRMie associating (water)" begin
        # Water with SAFTVRMie: 1 assoc pair → Val{1} analytical path.
        # Exercises VRMie I(Tr,ρr) _assoc_delta dispatch.
        model = SAFTVRMie(["water"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-4
    end

    @testset "HeterogcPCPSAFT associating (water)" begin
        # Water with HeterogcPCPSAFT: uses pack_assoc_params_gc (site_translator mapping)
        # and default g_hs _assoc_delta with bead-indexed assoc_icomp/jcomp.
        model = HeterogcPCPSAFT(["water"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-4
    end

    @testset "pharmaPCSAFT associating (water)" begin
        # pharmaPCSAFT uses a T-dependent sigma correction Δσh20(T) for water.
        # Tests that _assoc_delta override applies the correction so the
        # functional derivative at bulk matches the bulk chemical potential.
        model = pharmaPCSAFT(["water"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-4
    end

    @testset "SAFTgammaMie associating (water)" begin
        # Water with SAFTgammaMie: uses pack_assoc_params_gc + VRMie _assoc_delta
        # with params.meff for ρS and params.epsilon_species for Tr.
        model = SAFTgammaMie(["water"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ, [-10L, 10L], (3,))
        system = DFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-4
    end

    @testset "ePCSAFT electrolyte (bulk)" begin
        # ePCSAFT = pharmaPCSAFT (neutral) + hsdDH/ConstRSP (ions).
        # At uniform bulk density δFδρ_res ≈ μres/kT for all components.
        # ε_r from ConstRSP is a constant so the bulk pre-computation in
        # preallocate_params is exact.
        model = ePCSAFT(["water08"], ["sodium", "chloride"])

        z = [0.9, 0.05, 0.05]
        v = volume(model, p, T, z)
        ρbulk = z/v

        μ1 = Clapeyron.chemical_potential_res(model, p, T, z)/T/Clapeyron.Rgas()

        L = ClassicalDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], (3,))
        system = ElectrolyteDFTSystem(model, structure)
        ρ = ClassicalDFT.initialize_profiles(system)
        μ2 = ClassicalDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1, 1] rtol = 1e-4   # water08
        @test μ1[2] ≈ μ2[1, 2] rtol = 1e-4   # Na+
        @test μ1[3] ≈ μ2[1, 3] rtol = 1e-4   # Cl-
    end
end