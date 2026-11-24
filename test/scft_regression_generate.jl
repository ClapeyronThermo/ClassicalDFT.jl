# Prints the golden-value table for test_scft_regression.jl. Run only when the reference
# behaviour is meant to change:  julia --project test/scft_regression_generate.jl
include(joinpath(@__DIR__, "scft_regression_cases.jl"))

println("const SCFT_GOLDEN = Dict(")
for (name, f) in SCFT_REGRESSION_CASES
    t = @elapsed ρ = f()
    @info "$name: $(round(t, digits=1)) s, any NaN = $(any(isnan, ρ))"
    println("    \"$name\" => ", repr(scft_fingerprint(ρ)), ",")
end
println(")")
