# Figure for docs/src/tutorials/gpu_acceleration.md
# Reuses the existing CPU/GPU benchmark CSVs already committed under examples/ (columns:
# N, time, u_time) rather than re-running the CUDA benchmark. If you'd rather regenerate
# them fresh, run examples/benchmark_uniform.jl and examples/benchmark_uniform_gpu.jl first
# (they overwrite the same CSV filenames used below).
include("common.jl")
using CSV, DataFrames, CairoMakie

data = CSV.read("docs_figures/benchmark_data.csv", DataFrame)
N = data[:, 1]
t_metal = data[:, 2]
t_cpu = data[:, 3]
t_cuda = data[:, 4]

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="grid size N", ylabel="wall time / ms", yscale=log10, xscale=log10)
scatter!(ax, N, t_cpu, label="CPU (1 thread)")
scatter!(ax, N, t_cuda, label="CUDA")
scatter!(ax, N, t_metal, label="Metal")
axislegend(ax; position=:lt)
save(assetpath("gpu_acceleration_benchmark.png"), fig)

println("saved ", assetpath("gpu_acceleration_benchmark.png"))

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="grid size N", yscale=log10, xscale=log10)
scatter!(ax, N, t_cuda ./ t_cpu, label="CUDA / CPU")
scatter!(ax, N, t_metal ./ t_cpu, label="Metal / CPU")
axislegend(ax; position=:lt)
save(assetpath("gpu_acceleration_benchmark_comparison.png"), fig)

println("saved ", assetpath("gpu_acceleration_benchmark_comparison.png"))
