@testset "Morphology" begin
    # HeterogcPCPSAFT(["1-butanol"]) is already exercised (and known to work) in
    # test_coordinate_systems.jl; reused here purely as a real, GC-resolved model with
    # ≥2 distinct named groups, to exercise the core/matrix group-domain machinery without
    # depending on synthetic group parameters.
    model = HeterogcPCPSAFT(["1-butanol"])
    T = 298.15
    (p, vl, _) = Clapeyron.saturation_pressure(model, T)
    ρbulk = [1/vl]
    L = ClassicalDFT.length_scale(model)

    letters = String.(unique(first.(split.(model.groups.flattenedgroups, "_"))))
    @test length(letters) >= 2
    core = [letters[1]]

    # Dependency-free anti-correlation check (no Statistics in the test target).
    function _anticorrelated(a, b)
        a = vec(a); b = vec(b)
        ā, b̄ = sum(a)/length(a), sum(b)/length(b)
        return sum((a .- ā).*(b .- b̄)) < 0
    end

    function check_morphology(system)
        ρ = ClassicalDFT.initialize_profiles(system)
        @test all(ρ .> 0)

        nd = ClassicalDFT.dimension(system.structure)
        group_letters = first.(split.(model.groups.flattenedgroups, "_"))
        core_idx = findfirst(l -> l in core, group_letters)
        matrix_idx = findfirst(l -> !(l in core), group_letters)

        core_prof = selectdim(ρ, nd+1, core_idx)
        matrix_prof = selectdim(ρ, nd+1, matrix_idx)

        @test !all(core_prof .≈ core_prof[1])
        @test _anticorrelated(core_prof, matrix_prof)
        return ρ
    end

    @testset "LamellarStack1DCart" begin
        structure = ClassicalDFT.LamellarStack1DCart((p,T), ρbulk, [-5L, 5L], 51; core_groups=core)
        check_morphology(DFTSystem(model, structure))
    end

    @testset "LamellarStack2DCart" begin
        structure = ClassicalDFT.LamellarStack2DCart((p,T), ρbulk, [-5L 5L; -5L 5L], (21,21); core_groups=core)
        check_morphology(DFTSystem(model, structure))
    end

    @testset "LamellarStack3DCart" begin
        structure = ClassicalDFT.LamellarStack3DCart((p,T), ρbulk, [-5L 5L; -5L 5L; -5L 5L], (11,11,11); core_groups=core)
        check_morphology(DFTSystem(model, structure))
    end

    @testset "HexLattice2DCart" begin
        Lx = 5L
        bounds = [-Lx Lx; -sqrt(3)*Lx sqrt(3)*Lx]
        structure = ClassicalDFT.HexLattice2DCart((p,T), ρbulk, bounds, (11,19); core_groups=core)
        check_morphology(DFTSystem(model, structure))
    end

    @testset "HexLattice3DCart" begin
        Lx = 5L
        bounds = [-Lx Lx; -sqrt(3)*Lx sqrt(3)*Lx; -Lx Lx]
        structure = ClassicalDFT.HexLattice3DCart((p,T), ρbulk, bounds, (11,19,5); core_groups=core)
        check_morphology(DFTSystem(model, structure))
    end

    @testset "BCC3DCart" begin
        structure = ClassicalDFT.BCC3DCart((p,T), ρbulk, [-5L 5L; -5L 5L; -5L 5L], (11,11,11); core_groups=core)
        check_morphology(DFTSystem(model, structure))
    end

    @testset "Gyroid3DCart" begin
        structure = ClassicalDFT.Gyroid3DCart((p,T), ρbulk, [-5L 5L; -5L 5L; -5L 5L], (11,11,11); core_groups=core)
        check_morphology(DFTSystem(model, structure))
    end

    @testset "unknown core_groups errors clearly" begin
        structure = ClassicalDFT.BCC3DCart((p,T), ρbulk, [-5L 5L; -5L 5L; -5L 5L], (5,5,5); core_groups=["not_a_real_group"])
        @test_throws ErrorException ClassicalDFT.initialize_profiles(DFTSystem(model, structure))
    end
end
