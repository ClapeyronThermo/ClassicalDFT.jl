using Test, cDFT, FFTW

# Collapses the model/mol_structure/structure/system construction repeated across nearly
# every SCFT testset. `chain_specs` is a list of `(name, group_counts)` pairs, e.g.
# `[("diblock", ["A"=>5,"B"=>5])]`; the linear bead sequence for `mol_structure` is built
# automatically from each spec's group letters/counts, unless `mol_structure_strings`
# (a `Dict{String,String}` of raw `custom_structure` syntax, e.g. `"A(A)(A)A"`) is given
# to describe a branched (or otherwise non-default) topology instead.
function build_scft_system(chain_specs, chi; rho0=1.0, kappa=20.0, b=ones(size(chi,1)),
                            L=10.0, ngrid=65, rhobulk_seed=ones(length(chain_specs)),
                            ensemble=fill(:canonical, length(chain_specs)),
                            n_molecules=ones(length(chain_specs)),
                            structure=cDFT.Uniform1DCart((0.0,0.0), rhobulk_seed, [0.0,L], ngrid),
                            mol_structure_strings=nothing)
    model = cDFT.SCFTLatticeFluid(chain_specs, b, chi; rho0=rho0, kappa=kappa)
    mol_structure = if mol_structure_strings !== nothing
        Dict(name => cDFT.custom_structure(s) for (name, s) in mol_structure_strings)
    else
        Dict(name => cDFT.custom_structure(join(letter^n for (letter,n) in groups))
             for (name, groups) in chain_specs)
    end
    return cDFT.SCFTSystem(model, structure, cDFT.DFTOptions();
        mol_structure=mol_structure, ensemble=ensemble, n_molecules=n_molecules)
end

# FFT-based reference convolution, independent of cDFT's own `convolve!`/propagator
# machinery — used by the brute-force chain-integral checks below.
function _ref_conv(f::Vector{Float64}, kernel, ngrid::Int)
    P  = plan_rfft(zeros(ngrid), 1:1)
    iP = plan_irfft(zeros(ComplexF64, ngrid ÷ 2 + 1), ngrid, 1:1)
    return real.(iP * (kernel .* (P * f)))
end

@testset "SCFT" begin

    @testset "expand_model / get_species error paths" begin
        chi = [0.0 10.0; 10.0 0.0]
        model = cDFT.SCFTLatticeFluid([("diblock", ["A"=>3, "B"=>2])], [1.0, 1.2], chi; rho0=1.0, kappa=20.0)

        # a mol_structure inconsistent with the grouplist composition must throw
        bad_mol_structure = Dict("diblock" => cDFT.custom_structure("AABBB"))
        @test_throws AssertionError cDFT.expand_model(model, bad_mol_structure)

        # a branched mol_structure is now accepted (DiscreteGaussianChainPropagator walks
        # the resulting bond tree — see SCFTSpecies's docstring) rather than rejected.
        branched_model = cDFT.SCFTLatticeFluid([("chain", ["A"=>3, "B"=>2])], [1.0, 1.2], chi; rho0=1.0, kappa=20.0)
        branched_structure = Dict("chain" => cDFT.custom_structure("AA(B)AB"))
        expanded = cDFT.expand_model(branched_model, branched_structure)
        @test length(expanded.groups.flattenedgroups) == 5
        @test sum(expanded.groups.n_intergroups[1]) == 2 * 4  # 4 bonds, symmetric matrix
    end

    @testset "Propagator kernel matches analytic Gaussian-chain formula" begin
        L = 10.0
        b_A, b_B = 1.0, 2.0
        system = build_scft_system([("diblock", ["A"=>2, "B"=>2])], zeros(2,2);
            b=[b_A, b_B], L=L, ngrid=64, ensemble=[:canonical], n_molecules=[1.0])

        ν1 = 1.0 / L
        b_AB = sqrt((b_A^2 + b_B^2) / 2)
        for ((α, β), b_bond) in (((1,1), b_A), ((2,2), b_B), ((1,2), b_AB))
            kernel = system.propagator.kernel_map[(α, β)]
            @test real(kernel[1]) ≈ 1.0 atol = 1e-14
            @test real(kernel[2]) ≈ exp(-2π^2 * b_bond^2 * ν1^2 / 3) rtol = 1e-12
        end
    end

    @testset "Field formula matches closed form" begin
        chi = [0.0 10.0; 10.0 0.0]
        rho0 = 1.0
        kappa = 20.0
        system = build_scft_system([("chain", ["A"=>3, "B"=>2])], chi; rho0=rho0, kappa=kappa,
            ensemble=[:canonical], n_molecules=[10.0], ngrid=64)
        ngrid = system.structure.ngrid[1]

        ρ_A, ρ_B = 0.6, 0.4
        ρ = hcat(fill(ρ_A, ngrid), fill(ρ_B, ngrid))
        w = similar(ρ)
        cDFT.compute_fields!(system, ρ, w)

        # Expected: w_A = χ_AB/ρ₀ * ρ_B + ζ/ρ₀ * (ρ₊/ρ₀ - 1)
        ρ_total = ρ_A + ρ_B
        comp = (kappa / rho0) * (ρ_total / rho0 - 1.0)
        expected_wA = chi[1,2] / rho0 * ρ_B + comp
        expected_wB = chi[2,1] / rho0 * ρ_A + comp
        @test all(x -> isapprox(x, expected_wA; atol=1e-12), w[:, 1])
        @test all(x -> isapprox(x, expected_wB; atol=1e-12), w[:, 2])

        # The bulk (scalar) formula must agree with the spatial formula above
        w_bulk_check = cDFT.compute_bulk_fields(system.model, [ρ_A, ρ_B])
        @test w_bulk_check[1] ≈ expected_wA atol = 1e-12
        @test w_bulk_check[2] ≈ expected_wB atol = 1e-12

        # 3-species case, direct bulk-fields check (no full system needed)
        chi3 = [0.0 10.0 5.0; 10.0 0.0 3.0; 5.0 3.0 0.0]
        model3 = cDFT.SCFTLatticeFluid([("chain", ["A"=>1, "B"=>1, "S"=>1])], ones(3), chi3; rho0=rho0, kappa=kappa)
        bulk3 = [0.4, 0.3, 0.3]
        w_bulk3 = cDFT.compute_bulk_fields(model3, bulk3)
        ρ_total3 = sum(bulk3)
        comp3 = (kappa / rho0) * (ρ_total3 / rho0 - 1.0)
        @test w_bulk3[1] ≈ chi3[1,2]/rho0*bulk3[2] + chi3[1,3]/rho0*bulk3[3] + comp3
        @test w_bulk3[2] ≈ chi3[2,1]/rho0*bulk3[1] + chi3[2,3]/rho0*bulk3[3] + comp3
        @test w_bulk3[3] ≈ chi3[3,1]/rho0*bulk3[1] + chi3[3,2]/rho0*bulk3[2] + comp3
    end

    @testset "Uniform self-consistency" begin
        # A uniform system at bulk densities is exactly the fixed point of converge!'s
        # field map: compute_fields!(bulk) reproduces w_bulk (Δw=0 everywhere), so every
        # propagator value is exp(0)=1, Q̃=1, and the resulting density is bulk again.
        # Run through the public converge! entry point rather than hand-driving the
        # internal propagate!/compute_partition_functions/compute_densities! pipeline.
        system = build_scft_system([("diblock", ["A"=>5, "B"=>5])], [0.0 1.0; 1.0 0.0];
            ensemble=[:canonical], n_molecules=[1.0], ngrid=65)
        ngrid = system.structure.ngrid[1]

        bulk = cDFT.compute_bulk_densities(system)
        ρ = hcat(fill(bulk[1], ngrid), fill(bulk[2], ngrid))
        cDFT.converge!(system, ρ)

        @test all(x -> isapprox(x, bulk[1]; rtol=1e-6), ρ[:, 1])
        @test all(x -> isapprox(x, bulk[2]; rtol=1e-6), ρ[:, 2])
    end

    @testset "Free energy - uniform system" begin
        chi = [0.0 1.0; 1.0 0.0]
        rho0 = 1.0
        kappa = 20.0
        system = build_scft_system([("diblock", ["A"=>5, "B"=>5])], chi; rho0=rho0, kappa=kappa,
            ensemble=[:canonical], n_molecules=[1.0], ngrid=65)
        ngrid = system.structure.ngrid[1]

        bulk = cDFT.compute_bulk_densities(system)
        ρ = hcat(fill(bulk[1], ngrid), fill(bulk[2], ngrid))
        w_bulk = cDFT.compute_bulk_fields(system.model, bulk)
        w = hcat(fill(w_bulk[1], ngrid), fill(w_bulk[2], ngrid))
        # At the uniform fixed point Δw=0 everywhere, so every propagator value is
        # exp(0)=1 and Q̃=1 exactly (not just approximately) — no propagation needed.
        Q = [1.0]

        H = cDFT.free_energy(system, ρ, w, Q)

        dz = cDFT.structure_dz(system.structure)
        V_eff = cDFT.effective_volume(system, dz)
        U_int_expected = chi[1,2] * bulk[1] * bulk[2] * V_eff / rho0
        ρ_total = bulk[1] + bulk[2]
        U_comp_expected = (kappa / 2.0) * (ρ_total / rho0 - 1.0)^2 * V_eff
        wρ_sum_expected = (w_bulk[1]*bulk[1] + w_bulk[2]*bulk[2]) * V_eff
        w_bulk_sum = 5*w_bulk[1] + 5*w_bulk[2]  # 5 A-segments + 5 B-segments per chain
        molecule_sum_expected = -1.0 * (log(Q[1]) - w_bulk_sum)  # n_molecules=1.0
        H_expected = U_int_expected + U_comp_expected - wρ_sum_expected + molecule_sum_expected

        @test H ≈ H_expected rtol = 1e-8
    end

    @testset "Initialize profiles" begin
        system = build_scft_system([("chain", ["A"=>5, "B"=>5])], [0.0 10.0; 10.0 0.0];
            ensemble=[:canonical], n_molecules=[10.0], ngrid=64)
        ngrid = system.structure.ngrid[1]

        ρ = cDFT.initialize_profiles(system)
        @test size(ρ) == (ngrid, 2)
        @test all(ρ[:, 1] .== ρ[1, 1])
        @test all(ρ[:, 2] .== ρ[1, 2])

        ρ_pert = cDFT.initialize_profiles(system; noise=0.01)
        @test size(ρ_pert) == (ngrid, 2)
        @test !all(ρ_pert[:, 1] .== ρ_pert[1, 1])
    end

    @testset "Grand canonical solvent" begin
        # Same fixed-point argument as "Uniform self-consistency", extended to a
        # multi-molecule-type (canonical diblock + grand-canonical solvent) system,
        # run through the public converge! entry point.
        chi = [0.0 10.0 5.0; 10.0 0.0 3.0; 5.0 3.0 0.0]
        system = build_scft_system(
            [("diblock", ["A"=>5, "B"=>5]), ("solvent", ["S"=>1])], chi;
            ensemble=[:canonical, :grand_canonical], n_molecules=[10.0, 0.0],
            rhobulk_seed=[1.0, 0.3], ngrid=65)
        ngrid = system.structure.ngrid[1]

        bulk = cDFT.compute_bulk_densities(system)
        ρ = hcat((fill(b, ngrid) for b in bulk)...)
        cDFT.converge!(system, ρ)

        # Solvent density should stay at its bulk value
        @test all(x -> isapprox(x, bulk[3]; rtol=1e-6), ρ[:, 3])
    end

    @testset "Convergence - perturbed to uniform" begin
        # A symmetric AB diblock with weak χ should converge back to uniform
        # from a small perturbation (below the spinodal)
        chi = [0.0 0.5; 0.5 0.0]  # χN = 5, well below ODT (~10.5)
        system = build_scft_system([("diblock", ["A"=>5, "B"=>5])], chi;
            ensemble=[:canonical], n_molecules=[1.0], ngrid=65)

        bulk = cDFT.compute_bulk_densities(system)
        ρ = cDFT.initialize_profiles(system; noise=0.05)
        @test maximum(abs.(ρ[:, 1] .- bulk[1])) > 0.001  # verify it's actually perturbed

        cDFT.converge!(system, ρ)

        @test all(x -> isapprox(x, bulk[1]; rtol=1e-3), ρ[:, 1])
        @test all(x -> isapprox(x, bulk[2]; rtol=1e-3), ρ[:, 2])
    end

    @testset "Convergence - lamellar microphase separation" begin
        # Symmetric AB diblock above the ODT (χN=30 >> 10.5) should form lamellae
        N_seg = 20
        chi_val = 1.5  # χN = 30
        rho0 = 1.0
        chi = [0.0 chi_val; chi_val 0.0]

        # Box ≈ one lamellar period (Rg ≈ b√(N/6) ≈ 1.83, d ≈ 3.8 Rg ≈ 7)
        L = 7.0
        ngrid = 65
        n_chains = L / N_seg  # total segment density ≈ ρ₀
        system = build_scft_system([("diblock", ["A"=>N_seg÷2, "B"=>N_seg÷2])], chi;
            ensemble=[:canonical], n_molecules=[n_chains], L=L, ngrid=ngrid)

        bulk = cDFT.compute_bulk_densities(system)

        # Seed with sinusoidal perturbation to break symmetry toward lamellae
        ρ = cDFT.initialize_profiles(system)
        z = range(0, L, length=ngrid)
        amp = 0.1 * bulk[1]
        for i in 1:ngrid
            ρ[i, 1] += amp * cos(2π * z[i] / L)
            ρ[i, 2] -= amp * cos(2π * z[i] / L)
        end
        clamp!(ρ, 1e-10, Inf)

        cDFT.converge!(system, ρ)

        # Density profiles should be strongly non-uniform (microphase separated)
        amp_A = maximum(ρ[:, 1]) - minimum(ρ[:, 1])
        amp_B = maximum(ρ[:, 2]) - minimum(ρ[:, 2])
        @test amp_A > 0.5
        @test amp_B > 0.5

        # A and B should be out of phase (complementary domains)
        @test argmax(ρ[:, 1]) != argmax(ρ[:, 2])

        # Incompressibility: total density ≈ ρ₀ everywhere
        ρ_total = ρ[:, 1] .+ ρ[:, 2]
        @test all(x -> isapprox(x, rho0; rtol=0.05), ρ_total)
    end

    @testset "Morphology-seeded initialize_profiles" begin
        # A SCFTSystem built with a LamellarStack1DCart structure should have
        # initialize_profiles(system) reach the morphology-seeding dispatch branch
        # (src/structure/structure.jl), not SCFT's own flat bulk/perturbed seeding.
        L = 7.0
        ngrid = 33
        n_chains = L / 10
        structure = cDFT.LamellarStack1DCart((0.0, 0.0), [1.0], [0.0, L], ngrid; core_groups=["A"])
        system = build_scft_system([("diblock", ["A"=>5, "B"=>5])], [0.0 1.0; 1.0 0.0];
            ensemble=[:canonical], n_molecules=[n_chains], structure=structure)

        ρ = cDFT.initialize_profiles(system)

        # Morphology-seeded profile must be non-uniform (unlike SCFT's default flat seed)
        @test !all(ρ[:, 1] .== ρ[1, 1])
        @test !all(ρ[:, 2] .== ρ[1, 2])
        # A and B should be seeded out of phase (core_groups=["A"] vs. the rest)
        @test argmax(ρ[:, 1]) != argmax(ρ[:, 2])
        # Strictly positive everywhere (amplitude < 1 guarantee from _fill_morphology!)
        @test all(ρ .> 0)
    end

    @testset "Branched architecture: propagator matches an independent reference sweep" begin
        # Second, independently-written implementation of the same bottom-up/top-down
        # tree recursion _dgc_tree_sweep! (src/propagator/discrete_gaussian_chain.jl)
        # implements — own convolution routine, own loops/data structures — used to
        # catch indexing/implementation regressions in the real propagator. Covers a
        # plain linear chain (the degenerate case — every existing test above already
        # exercises this, but only indirectly via full SCF convergence; this checks the
        # propagator sweep itself in isolation), a symmetric 3-arm star (a genuine
        # multi-child branch point), and an asymmetric 2-species Y-shaped chain.
        function ref_tree_sweep(seg_spec, local_bonds, lev, kernels, ef, ngrid)
            Nc = length(seg_spec)
            n_levels = maximum(lev)
            i_root = findfirst(==(1), lev)
            q_in  = [zeros(ngrid) for _ in 1:Nc]
            q_out = [zeros(ngrid) for _ in 1:Nc]
            for L in n_levels:-1:1
                for k in findall(==(L), lev)
                    acc = copy(ef[seg_spec[k]])
                    for j in findall(local_bonds[k, :] .& (lev .== L + 1))
                        acc .*= _ref_conv(q_in[j], kernels[minmax(seg_spec[k], seg_spec[j])], ngrid)
                    end
                    q_in[k] = acc
                end
            end
            q_out[i_root] = copy(ef[seg_spec[i_root]])
            for L in 1:n_levels-1
                for k in findall(==(L), lev)
                    children = findall(local_bonds[k, :] .& (lev .== L + 1))
                    conv_to_k = Dict(s => _ref_conv(q_in[s], kernels[minmax(seg_spec[k], seg_spec[s])], ngrid) for s in children)
                    for j in children
                        base = copy(q_out[k])
                        for s in children
                            s == j && continue
                            base .*= conv_to_k[s]
                        end
                        q_out[j] = _ref_conv(base, kernels[minmax(seg_spec[k], seg_spec[j])], ngrid) .* ef[seg_spec[j]]
                    end
                end
            end
            return q_in, q_out, i_root
        end

        function check_topology(mol_structure_string, group_counts; L=10.0, ngrid=24, n_molecules=2.0)
            model_species = unique(first.(group_counts))
            b = ones(length(model_species))
            chi = zeros(length(model_species), length(model_species))
            system = build_scft_system([("mol", group_counts)], chi; b=b, L=L, ngrid=ngrid,
                ensemble=[:canonical], n_molecules=[n_molecules],
                mol_structure_strings=Dict("mol" => mol_structure_string))

            species = system.species
            nspecies = length(model_species)
            z = range(0, L, length=ngrid+1)[1:end-1]
            # A distinct, non-uniform field per species so every bond direction is exercised.
            w = hcat((0.3 .* sin.(2π .* (α) .* z ./ L) .+ 0.05 .* α for α in 1:nspecies)...)
            w_bulk = cDFT.compute_bulk_fields(system.model, cDFT.compute_bulk_densities(system))
            ef = [exp.(w_bulk[α] .- w[:, α]) for α in 1:nspecies]

            seg_spec = species.sequence[1]
            ig = species.i_groups[1]
            lev = species.levels[ig]
            local_bonds = species.n_intergroups[1][ig, ig] .!= 0
            # Reuse cDFT's own R2C (half-complex) kernels directly — _ref_conv expects
            # the same half-complex kernel shape _dgc_tree_sweep! uses.
            kernels = system.propagator.kernel_map

            q_in_ref, q_out_ref, i_root = ref_tree_sweep(seg_spec, local_bonds, lev, kernels, ef, ngrid)

            ρ = ones(ngrid, nspecies)
            cache_propagator = cDFT.preallocate_propagator(system, system.propagator, ρ, system.options.device)
            q_in, q_out, buf_r, buf_c, child_buf, P, iP = cache_propagator
            cDFT.propagate!(system, ρ, w, cache_propagator; w_bulk=w_bulk, exp_field=nothing)

            for k in 1:length(seg_spec)
                @test q_in[1][:, k]  ≈ q_in_ref[k]  rtol=1e-10
                @test q_out[1][:, k] ≈ q_out_ref[k] rtol=1e-10
            end

            V_eff = L
            Q = cDFT.compute_partition_functions(system, w, w_bulk, q_in, cDFT.structure_dz(system.structure); V_eff=V_eff)
            Q_ref = sum(q_in_ref[i_root]) * (L/ngrid) / V_eff
            @test Q[1] ≈ Q_ref rtol=1e-10

            ρ_actual = similar(ρ)
            cDFT.compute_densities!(system, w, w_bulk, q_in, q_out, Q, ρ_actual; V_eff=V_eff)
            inv_ef = [exp.(w[:, α] .- w_bulk[α]) for α in 1:nspecies]
            prefactor = n_molecules / (V_eff * Q_ref)
            ρ_ref = zeros(ngrid, nspecies)
            for k in 1:length(seg_spec)
                α = seg_spec[k]
                ρ_ref[:, α] .+= prefactor .* q_in_ref[k] .* q_out_ref[k] .* inv_ef[α]
            end
            @test ρ_actual ≈ ρ_ref rtol=1e-10
        end

        # Degenerate case: a plain linear chain, routed through the same tree code path
        # every other case here uses (there is no separate "linear" code path anymore).
        check_topology("AAAAA", ["A"=>5])

        # Symmetric 3-arm star: a genuine multi-child branch point at the root.
        check_topology("A(A)(A)A", ["A"=>4])

        # Asymmetric, 2-species Y-shaped chain: branch point away from either species'
        # own bond kernel being trivially self-similar.
        check_topology("AA(B)AB", ["A"=>3, "B"=>2])
    end

    @testset "Branched architecture: end-to-end convergence (3-arm star melt)" begin
        # Golden-path smoke test: a single-species homopolymer star melt should converge
        # to a uniform density equal to rho0, exactly like a linear homopolymer melt would
        # (branching doesn't change the bulk equation of state for a single species with
        # no chi interactions — only the single-chain statistics, which don't appear in a
        # spatially-uniform solution).
        rho0 = 1.0
        L = 10.0
        n_seg = 7
        # n_molecules chosen so n_seg segments/chain * n_molecules/L exactly fills to rho0
        # (matching how e.g. the "Uniform self-consistency" testset above tunes its own
        # n_molecules so nbeads*n_molecules/L == rho0).
        system = build_scft_system([("star", ["A"=>n_seg])], zeros(1,1); rho0=rho0, kappa=20.0,
            ensemble=[:canonical], n_molecules=[L/n_seg], L=L, ngrid=33,
            mol_structure_strings=Dict("star" => "AAA(AAA)A"))

        bulk = cDFT.compute_bulk_densities(system)
        ngrid = system.structure.ngrid[1]
        ρ = fill(bulk[1], ngrid, 1)
        cDFT.converge!(system, ρ)

        @test all(x -> isapprox(x, bulk[1]; rtol=1e-6), ρ[:, 1])
        @test isapprox(bulk[1], rho0; rtol=1e-6)
    end
end
