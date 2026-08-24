module SciMLBaseClassicalDFTExt
    import ClassicalDFT
    import SciMLBase
    import KernelAbstractions as KA
    import FFTW


    ################### DYNAMIC DENSITY FUNCTIONAL THEORY ###################

    function SciMLBase.ODEProblem(system::ClassicalDFT.AbstractClassicalDFTSystem, ρ, tspan, kwargs...)
        FP = ClassicalDFT.fptype(system.options)
        ngrid    = system.structure.ngrid
        nd       = length(ngrid)
        L        = FP(ClassicalDFT.length_scale(system.model))
        k        = ClassicalDFT.structure_ω(system.structure, system.options.device, FP) .* L

        map_grad =  2(π .* k .* im)
        _2π² = FP((2π)^2)
        map_lapl = dropdims(-_2π².* sum(k.^2, dims=nd+1), dims=nd+1)

        tmp      = KA.allocate(system.options.device, FP, ngrid...)
        buf      = KA.allocate(system.options.device, Complex{FP}, ngrid...)
        buf_real = similar(tmp, FP)
        # ρ        = similar(ρ0, FP)
        P        = FFTW.plan_fft!(buf)
        iP       = FFTW.plan_ifft!(buf)
        
        μ, cache_model, cache_external_field, cache_propagator = ClassicalDFT.preallocate(system, ρ)

        function ddft_rhs_log!(dη, η, params, t)
            @. ρ = exp(clamp(η, -50, 30))
            # println(t, " ", minimum(ρ), " ", maximum(ρ))
            # println(t, " ", minimum(μ), " ", maximum(μ))

            ClassicalDFT.δFδρ_res!(system, ρ, μ, cache_model...)
            # println(t, " ", minimum(μ), " ", maximum(μ))
            ClassicalDFT.evaluate_external_field!(system, ρ, μ, cache_external_field)
            ClassicalDFT.propagate!(system, ρ, μ, cache_propagator)

            for α in axes(η, nd + 1)
                η_α  = selectdim(η,  nd + 1, α)
                μ_α  = selectdim(μ,  nd + 1, α)
                dη_α = selectdim(dη, nd + 1, α)

                # Laplacians — initialise dη with ∇²η + ∇²μ                
                ClassicalDFT.convolve!(dη_α, η_α, map_lapl, P, iP, buf)
                ClassicalDFT.convolve!(tmp,  μ_α, map_lapl, P, iP, buf)
                @. dη_α += tmp

                # Gradient terms — accumulate directly
                for d in 1:nd
                    ClassicalDFT.convolve!(tmp, η_α, selectdim(map_grad, nd+1, d), P, iP, buf)   # tmp = ∇ηd
                    @. dη_α += tmp^2                                # |∇η|²

                    ClassicalDFT.convolve!(buf_real, μ_α, selectdim(map_grad, nd+1, d), P, iP, buf)  # buf_real = ∇μd
                    @. dη_α += buf_real * tmp                          # ∇μ·∇η
                end
            end
        end
        return SciMLBase.ODEProblem(ddft_rhs_log!, log.(ρ), tspan, system, kwargs...)
    end

    ######### ALTERNATE CONVERGE! IMPLEMENTATION FOR DDFT STEADY STATES #########
    # function ClassicalDFT.converge!(system::ClassicalDFT.AbstractClassicalDFTSystem, ρ, alg::SciMLBase.AbstractODEAlgorithm)
    #     cb = TerminateSteadyState(abstol=1e-4, reltol=1e-4)
    #     prob = SciMLBase.ODEProblem(system, log.(ρ), (0.0, 1e6), callback=cb)
    #     sol = DifferentialEquations.solve(prob, alg, save_everystep=false, save_start=false)
    #     ρ .= exp.(sol[end])
    # end
end