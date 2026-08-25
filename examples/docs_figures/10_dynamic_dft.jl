# Figures for docs/src/tutorials/dynamic_dft.md
include("common.jl")
using Clapeyron, ClassicalDFT, OrdinaryDiffEqStabilizedRK, CairoMakie

model = PCSAFT(["water", "hexane"])
p, T = 1e5, 290.15

x, n, _ = tp_flash(model, p, T, [0.5, 0.5], MichelsenTPFlash(equilibrium=:lle, K0=[1000.0, 1e-4]))
ρ1 = x[1,:] ./ Clapeyron.volume(model, p, T, x[1,:])
ρ2 = x[2,:] ./ Clapeyron.volume(model, p, T, x[2,:])
ρb = (ρ1 .+ ρ2) ./ 2

L = ClassicalDFT.length_scale(model)
ngrid = 51
structure = ClassicalDFT.Uniform2DCart((p, T), ρb, [-10L 10L; -10L 10L], (ngrid, ngrid))
system = DFTSystem(model, structure)

ρ0 = ClassicalDFT.initialize_profiles(system; noise=0.01)
println("initialized profiles with noise=0.01")

prob = ODEProblem(system, ρ0, (0.0, 1e1))
sol = solve(prob, ROCK2())

t0, tend = sol.t[1], sol.t[end]
fig = plot(system, exp.(sol(t0 + 0.05*(tend - t0))))
# set ylims based on ρ1 and ρ2
ylims = (0.9*minimum([ρ1; ρ2]), 1.1*maximum([ρ1; ρ2]))
fig
save(assetpath("dynamic_dft_early.png"), fig)
fig, ax, _ = plot(system, exp.(sol(t0 + 0.3*(tend - t0))))
Makie.ylims!(ax, ylims)
save(assetpath("dynamic_dft_mid.png"), fig)
fig, ax, _ = plot(system, exp.(sol(tend)))
Makie.ylims!(ax, ylims)
save(assetpath("dynamic_dft_late.png"), fig)

println("saved dynamic_dft_{early,mid,late}.png to ", ASSETS)
