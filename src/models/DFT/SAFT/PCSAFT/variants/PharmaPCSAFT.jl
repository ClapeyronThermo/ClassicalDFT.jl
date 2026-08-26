import Clapeyron: pharmaPCSAFTModel, Δσh20, water08_k

"""
    pharmaPCSAFT(components::Vector{String})

A PC-SAFT variant aimed at pharmaceutical/aqueous systems (Held et al.), which applies a temperature-dependent correction to the water segment diameter and temperature-dependent binary corrections to the dispersion energy cross-parameters. As with `PCSAFT`, our DFT implementation uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron.
"""
pharmaPCSAFT

function preallocate_params(system::DFTSystem{<:pharmaPCSAFTModel})
    params_base, nc = invoke(preallocate_params, Tuple{DFTSystem{<:PCSAFTModel}}, system)

    model       = system.model
    backend     = system.options.device
    FP          = fptype(system.options)
    temperature = system.structure.conditions[2]

    k  = Int(water08_k(model))
    nc_model = length(model)
    sigma_eff = copy(model.params.sigma.values)

    if k > 0
        Δσ = Δσh20(temperature)
        for i in 1:nc_model, j in 1:nc_model
            sigma_eff[i,j] += (0.5*(k == i) + 0.5*(k == j)) * Δσ
        end
    end

    # Reduced units: the raw effective diameter above (base σ + temperature-dependent
    # water shift) must be divided by L, same as the `sigma`/`HSd` params_base already
    # inherits from PCSAFT.jl's preallocate_params via `invoke` above — otherwise this
    # overwrite would silently undo that fix. Compute the shift in raw units first (as
    # above), then reduce the whole thing once, matching PCSAFT.jl's convention.
    L = length_scale(model)
    sigma_eff = sigma_eff ./ L

    # pharmaPCSAFT stores raw LB cross-terms (no k_ij baked in); Clapeyron's m2ϵσ3
    # applies (1 - k0[i,j] - k1[i,j]*T) at runtime. Pre-apply here so f_disp is correct.
    epsilon_eff = copy(model.params.epsilon.values)
    k0_mat = model.params.k.values
    k1_mat = model.params.kT.values
    for i in 1:nc_model
        for j in 1:nc_model
            i == j && continue
            epsilon_eff[i,j] *= (1.0 - k0_mat[i,j] - k1_mat[i,j]*temperature)
        end
    end

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        eps_vals = model.params.epsilon_assoc.values
        sig3_eff = [begin
                        i, j, _, _ = idx_to_ijab(eps_vals, idx)
                        sigma_eff[i, j]^3
                    end
                    for idx in 1:length(eps_vals.values)]
        return merge(params_base, (;
            sigma      = adapt_to_device(backend, FP, sigma_eff),
            epsilon    = adapt_to_device(backend, FP, epsilon_eff),
            assoc_sig3 = adapt_to_device(backend, FP, sig3_eff),
        )), nc
    else
        return merge(params_base, (;
            sigma   = adapt_to_device(backend, FP, sigma_eff),
            epsilon = adapt_to_device(backend, FP, epsilon_eff),
        )), nc
    end
end