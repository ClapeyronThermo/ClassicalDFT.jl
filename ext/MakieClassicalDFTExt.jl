module MakieClassicalDFTExt

using ClassicalDFT
using Adapt
using Makie

_maybe_texlabel(s, latex::Bool; upright::Bool=false) = latex ? ClassicalDFT.texlabel(s; upright=upright) : s

# Digits are already upright in math mode, so plain `texlabel` (no `upright`) is fine for
# tick numbers -- this just gets them onto the same LaTeX/MathTeXEngine font as every other
# labelled element instead of Makie's own default tick-label font.
function _format_tick(v::Real)
    s = string(v)
    return endswith(s, ".0") ? s[1:end-2] : s
end
_latex_tickformat(latex::Bool) = latex ? (values -> ClassicalDFT.texlabel.(_format_tick.(values))) : Makie.automatic

# Every geometry method below returns `Makie.FigureAxisPlot(fig, ax, plt)` rather than a bare
# `Figure` -- it's still directly `save()`-able (FigureAxisPlot <: Makie.FigureLike) so
# `save(path, plot(system, ρ))` one-liners keep working unchanged, but callers can now also
# do `fig, ax, plt = plot(system, ρ)` to get a handle on the Axis for post-hoc edits (limits,
# title, legend, annotations, ...). When a call draws multiple curves/heatmaps (one per
# species/group), `plt` is only the last one drawn -- `FigureAxisPlot.plot` is typed as a
# single `AbstractPlot`, it can't hold all of them. Every plot object is still reachable via
# `ax.scene.plots`, in the same draw order as the internal `groups` list, if a specific one
# (not the last) is needed.

# `segment`/`size` convert a number-density profile to a dimensionless volume fraction
# for segment-based (SAFT-style) models. SCFT's `SCFTLatticeFluid` has no such params —
# its converged profiles are already volume fractions — so those models get norm_const=1.
function _norm_const(species, model, i::Int, k::Int)
    hasproperty(model.params, :segment) || return 1.0
    return species.nbeads[i] > 1 ? model.params.segment[k]*species.size[k]^3*ClassicalDFT.N_A : model.params.segment[i]*species.size[i]^3*ClassicalDFT.N_A
end

_normalized_ylabel(model) = hasproperty(model.params, :segment) ? "ρσ³" : "φ"

# `equilibrium_densities`, if given, is `(ρ_phase1, ρ_phase2)` -- the two bulk coexisting
# number densities of a two-phase-separating system, one entry per component, in the same
# convention a `tp_flash`/`saturation_pressure` call already returns (e.g. `ρ1`, `ρ2` in
# dynamic_dft.jl). When given, a heatmap/volume group's color range is pinned to those bulk
# values (scaled the same way its plotted density is, via `_norm_const`) instead of that
# frame's own `extrema` -- so a movie of frames plotted with the same `equilibrium_densities`
# gets one consistent color bar throughout, rather than one that rescales itself every frame
# based on how far that frame's domain has phase-separated so far.
function _group_colorrange(equilibrium_densities, species, model, members)
    equilibrium_densities === nothing && return nothing
    ρ_phase1, ρ_phase2 = equilibrium_densities
    v1 = sum(_norm_const(species, model, i, k) * ρ_phase1[i] for (i, k) in members) / length(members)
    v2 = sum(_norm_const(species, model, i, k) * ρ_phase2[i] for (i, k) in members) / length(members)
    return v1 <= v2 ? (v1, v2) : (v2, v1)
end

# ── Aggregation ("plot_by") / coloring ("color_by") ──────────────────────
#
# Both are ∈ (:bead, :group, :molecule), from finest to coarsest granularity. `:bead`
# matches today's default (one curve per flattened bead index, e.g. "A_1"/"A_2"/"B_1");
# `:group` averages instances of the same named group within a component together (e.g.
# all "A_*" beads into one "A" curve); `:molecule` averages every bead of a component into
# one curve. `color_by` controls color assignment only, independent of aggregation, but
# can't be finer than `plot_by` (no per-bead data survives once beads are averaged
# away).

const _LEVEL_RANK = (bead = 1, group = 2, molecule = 3)

function _check_profile_color_by(plot_by::Symbol, color_by::Symbol)
    haskey(_LEVEL_RANK, plot_by) || error("plot_by must be :bead, :group or :molecule, got :$plot_by")
    haskey(_LEVEL_RANK, color_by) || error("color_by must be :bead, :group or :molecule, got :$color_by")
    _LEVEL_RANK[color_by] >= _LEVEL_RANK[plot_by] || error(
        "color_by=:$color_by cannot be finer-grained than plot_by=:$plot_by " *
        "(granularity order: bead < group < molecule) — there's no per-$(color_by) data " *
        "left once beads have been averaged to the :$plot_by level.")
end

function _group_key(species, model, i::Int, k::Int, level::Symbol)
    if level === :molecule
        return (i,)
    elseif level === :group
        return species.nbeads[i] > 1 ? (i, ClassicalDFT._group_letter(model.groups.flattenedgroups[k])) : (i,)
    else # :bead
        return (i, k)
    end
end

function _profile_label(species, model, i::Int, k::Int, level::Symbol)
    species_name = model.components[i]
    if level === :molecule || species.nbeads[i] == 1
        return species_name
    elseif level === :group
        return "$species_name $(ClassicalDFT._group_letter(model.groups.flattenedgroups[k]))"
    else # :bead
        return "$species_name $(model.groups.flattenedgroups[k])"
    end
end

# One entry per curve/field to draw: (label, [(i,k) beads to average together]).
function _plot_groups(species, model, plot_by::Symbol)
    members = Dict{Any,Vector{Tuple{Int,Int}}}()
    order = Any[]
    for i in ClassicalDFT.@comps
        for k in ClassicalDFT.@chain(i)
            key = _group_key(species, model, i, k, plot_by)
            if !haskey(members, key)
                members[key] = Tuple{Int,Int}[]
                push!(order, key)
            end
            push!(members[key], (i, k))
        end
    end
    return [(_profile_label(species, model, members[key][1]..., plot_by), members[key]) for key in order]
end

# Dict{color_key,color}, assigned in first-encountered order from `palette` (defaults to
# ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS -- override via the `color_scheme` kwarg on `plot(...)`).
function _assign_colors(color_keys, palette=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS)
    colors = Dict{Any,Any}()
    idx = 0
    for key in color_keys
        if !haskey(colors, key)
            idx += 1
            colors[key] = palette[mod1(idx, length(palette))]
        end
    end
    return colors
end

# For each (label, members) group from `_plot_groups`, its assigned color (keyed at
# `color_by` granularity, which may be coarser than `plot_by` so several groups can
# share one color).
function _group_colors(groups, species, model, color_by::Symbol, palette=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS)
    color_keys = [_group_key(species, model, members[1]..., color_by) for (_, members) in groups]
    colors = _assign_colors(color_keys, palette)
    return [colors[key] for key in color_keys]
end

# `only`, if given, restricts drawing to specific group(s) -- matched against the exact label
# text that would otherwise show up in the legend (from `_profile_label`/`_plot_groups`), so it
# lines up with whatever `plot_by` granularity is in effect (:bead/:group/:molecule) without
# the caller needing to know species/bead indices. Filters `groups`/`colors` together, applied
# *after* `_group_colors` assigns from the full unfiltered list, so a retained group keeps the
# same color it would have had alongside the others (e.g. "water" stays the same blue whether
# plotted with hexane or on its own).
function _filter_groups(groups, colors, only)
    only === nothing && return groups, colors
    wanted = only isa AbstractString ? (only,) : Tuple(only)
    keep = [i for (i, (label, _)) in enumerate(groups) if label in wanted]
    isempty(keep) && error("only=$only matched no plotted group; available labels are $(first.(groups))")
    return groups[keep], colors[keep]
end

# Figure kwargs shared by every geometry's `Makie.plot` method below: pixel size (from
# `width ∈ (:single,:double)` at `dpi`, matching the rcParams `figure.figsize`/`figure.dpi`
# pair) and background color (`figure.facecolor="white"`). `font` (a font family name,
# e.g. "Arial"; `nothing` = Makie's own default) is applied via `Figure(fonts=(;regular=...))`
# rather than a global `Theme`, so it stays a per-call option with no persistent state.
function _cdft_figure(width::Symbol, dpi::Real, font)
    sz = ClassicalDFT.classicaldft_figure_size(width, dpi)
    return font === nothing ? Figure(size=sz, backgroundcolor=:white) :
                               Figure(size=sz, backgroundcolor=:white, fonts=(; regular=font))
end

function Makie.plot(system::ClassicalDFT.AbstractClassicalDFTSystem, profiles; x_units=:normalized, y_units=:normalized, latex=false, plot_by=:bead, color_by=:bead, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false, equilibrium_densities=nothing, only=nothing)
    return Makie.plot(system, system.structure, profiles; x_units=x_units, y_units=y_units, latex=latex, plot_by=plot_by, color_by=color_by, color_scheme=color_scheme, font=font, width=width, dpi=dpi, grid=grid, equilibrium_densities=equilibrium_densities, only=only)
end

function Makie.plot(system::ClassicalDFT.AbstractClassicalDFTSystem, structure::ClassicalDFT.DFTStructure{1,ClassicalDFT.Cartesian,M}, profiles; x_units=:normalized, y_units=:mass, latex=false, plot_by=:bead, color_by=:bead, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false, equilibrium_densities=nothing, only=nothing) where M
    # `equilibrium_densities` only applies to the 2D/3D heatmap/volume methods below -- accepted
    # (and ignored) here too so the generic `Makie.plot(system, profiles; ...)` entry point can
    # forward it unconditionally regardless of the system's structure dimensionality.
    _check_profile_color_by(plot_by, color_by)
    structure = system.structure
    model = system.model
    if model isa ClassicalDFT.ElectrolyteModel
        model = model.neutralmodel
    end
    species = system.species

    bounds = structure.bounds
    z = ClassicalDFT.uniform_range(structure, 1)
    L = ClassicalDFT.length_scale(model)
    _ρ = Adapt.adapt(CPU(), profiles)

    fig = _cdft_figure(width, dpi, font)

    ax = Axis(fig[1, 1];
        xgridvisible=grid, ygridvisible=grid,
        xgridcolor=ClassicalDFT.CLASSICALDFT_GRID_COLOR, ygridcolor=ClassicalDFT.CLASSICALDFT_GRID_COLOR,
        xticklabelsize=ClassicalDFT.CLASSICALDFT_TICK_LABELSIZE, yticklabelsize=ClassicalDFT.CLASSICALDFT_TICK_LABELSIZE,
        xlabelsize=ClassicalDFT.CLASSICALDFT_AXES_LABELSIZE, ylabelsize=ClassicalDFT.CLASSICALDFT_AXES_LABELSIZE,
        xtickformat=_latex_tickformat(latex), ytickformat=_latex_tickformat(latex))

    if x_units == :normalized
        X = z./L
    elseif x_units == :angstrom
        X = z.*1e10
    elseif x_units == :nanometer
        X = z.*1e9
    else
        X = z
    end

    function bead_Y(i, k)
        norm_const = _norm_const(species, model, i, k)
        if y_units == :normalized
            return _ρ[:,k].*norm_const
        elseif y_units == :mass
            Mw = model.params.Mw[k]
            return _ρ[:,k].*Mw/1e3
        elseif y_units == :angstrom
            return _ρ[:,k].*ClassicalDFT.N_A/1e30
        else
            return _ρ[:,k]
        end
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by, color_scheme)
    groups, colors = _filter_groups(groups, colors, only)

    ymax = 0.
    plt = nothing

    for ((label, members), c) in zip(groups, colors)
        Y = sum(bead_Y(i,k) for (i,k) in members) ./ length(members)
        plt = Makie.lines!(ax, X, Y; label=_maybe_texlabel(label,latex; upright=true), linewidth=3, color=c)
        ymax = max(ymax,maximum(Y))
    end

    if x_units == :normalized
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]).*1e9)
        x_norm = "nm"
    else
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]))
        x_norm = "m"
    end

    Makie.ylims!(ax,(0,1.1*ymax))
    ax.xlabel = _maybe_texlabel("z / "*x_norm,latex)

    if y_units == :normalized
        ax.ylabel = _maybe_texlabel(_normalized_ylabel(model),latex)
    elseif y_units == :mass
        ax.ylabel = _maybe_texlabel("ρ / (kg/m³)",latex)
    else
        ax.ylabel = _maybe_texlabel("ρ / (mol/m³)",latex)
    end

    Makie.axislegend(ax; position=:lt, framevisible=false, labelsize=ClassicalDFT.CLASSICALDFT_LEGEND_FONTSIZE)

    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.plot(system::ClassicalDFT.AbstractClassicalDFTSystem, structure::Union{ClassicalDFT.DFTStructure{1,ClassicalDFT.Spherical,M},ClassicalDFT.DFTStructure{1,ClassicalDFT.Cylindrical,M}}, profiles; x_units=:normalized, y_units=:mass, latex=false, plot_by=:bead, color_by=:bead, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false, equilibrium_densities=nothing, only=nothing) where M
    # `equilibrium_densities` only applies to the 2D/3D heatmap/volume methods below -- see the
    # Cartesian 1D method just above for why it's accepted (and ignored) here too.
    _check_profile_color_by(plot_by, color_by)
    structure = system.structure
    model = system.model
    if model isa ClassicalDFT.ElectrolyteModel
        model = model.neutralmodel
    end
    species = system.species

    bounds = structure.bounds
    z = ClassicalDFT.structure_r(structure)
    L = ClassicalDFT.length_scale(model)

    _ρ = Adapt.adapt(CPU(), profiles)

    fig = _cdft_figure(width, dpi, font)
    ax = Axis(fig[1, 1];
        xgridvisible=grid, ygridvisible=grid,
        xgridcolor=ClassicalDFT.CLASSICALDFT_GRID_COLOR, ygridcolor=ClassicalDFT.CLASSICALDFT_GRID_COLOR,
        xticklabelsize=ClassicalDFT.CLASSICALDFT_TICK_LABELSIZE, yticklabelsize=ClassicalDFT.CLASSICALDFT_TICK_LABELSIZE,
        xlabelsize=ClassicalDFT.CLASSICALDFT_AXES_LABELSIZE, ylabelsize=ClassicalDFT.CLASSICALDFT_AXES_LABELSIZE,
        xtickformat=_latex_tickformat(latex), ytickformat=_latex_tickformat(latex))

    if x_units == :normalized
        X = z./L
    elseif x_units == :angstrom
        X = z.*1e10
    elseif x_units == :nanometer
        X = z.*1e9
    else
        X = z
    end

    function bead_Y(i, k)
        norm_const = _norm_const(species, model, i, k)
        if y_units == :normalized
            return _ρ[:,k].*norm_const
        elseif y_units == :mass
            Mw = model.params.Mw[k]
            return _ρ[:,k].*Mw/1e3
        elseif y_units == :angstrom
            return _ρ[:,k].*ClassicalDFT.N_A/1e30
        else
            return _ρ[:,k]
        end
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by, color_scheme)
    groups, colors = _filter_groups(groups, colors, only)

    ymax = 0.
    plt = nothing
    for ((label, members), c) in zip(groups, colors)
        Y = sum(bead_Y(i,k) for (i,k) in members) ./ length(members)
        plt = Makie.lines!(ax, X, Y; label=_maybe_texlabel(label,latex; upright=true), linewidth=3, color=c)
        ymax = max(ymax,maximum(Y))
    end

    if x_units == :normalized
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]).*1e9)
        x_norm = "nm"
    else
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]))
        x_norm = "m"
    end

    Makie.ylims!(ax,(0,1.1*ymax))
    ax.xlabel = _maybe_texlabel("r / "*x_norm,latex)

    if y_units == :normalized
        ax.ylabel = _maybe_texlabel(_normalized_ylabel(model),latex)
    elseif y_units == :mass
        ax.ylabel = _maybe_texlabel("ρ / (kg/m³)",latex)
    else
        ax.ylabel = _maybe_texlabel("ρ / (mol/m³)",latex)
    end

    Makie.axislegend(ax; position=:lt, framevisible=false, labelsize=ClassicalDFT.CLASSICALDFT_LEGEND_FONTSIZE)

    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.plot(system::Union{ClassicalDFT.DFTSystem,ClassicalDFT.DGTSystem,ClassicalDFT.SCFTSystem}, structure::ClassicalDFT.DFTStructure{2,ClassicalDFT.Cartesian,M}, profiles; x_units=:normalized, y_units=:normalized, latex=false, plot_by=:bead, color_by=:bead, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false, equilibrium_densities=nothing, only=nothing) where M
    _check_profile_color_by(plot_by, color_by)
    structure = system.structure
    model = system.model
    species = system.species

    bounds = structure.bounds

    _ρ = Adapt.adapt(CPU(), profiles)
    x = ClassicalDFT.uniform_range(structure,1)
    y = ClassicalDFT.uniform_range(structure,2)
    L = ClassicalDFT.length_scale(model)

    fig = _cdft_figure(width, dpi, font)
    ax = Axis(fig[1, 1];
        xgridvisible=grid, ygridvisible=grid,
        xgridcolor=ClassicalDFT.CLASSICALDFT_GRID_COLOR, ygridcolor=ClassicalDFT.CLASSICALDFT_GRID_COLOR,
        xtickformat=_latex_tickformat(latex), ytickformat=_latex_tickformat(latex),
        aspect=Makie.DataAspect())

    if x_units == :normalized
        X = x./L
    elseif x_units == :angstrom
        X = x.*1e10
    elseif x_units == :nanometer
        X = x.*1e9
    else
        X = x
    end

    if y_units == :normalized
        Y = y./L
    elseif y_units == :angstrom
        Y = y.*1e10
    elseif y_units == :nanometer
        Y = y.*1e9
    else
        Y = y
    end

    function bead_Z(i, k)
        norm_const = _norm_const(species, model, i, k)
        return _ρ[:,:,k].*norm_const
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by, color_scheme)
    groups, colors = _filter_groups(groups, colors, only)

    plt = nothing
    for ((label, members), c) in zip(groups, colors)
        Z = sum(bead_Z(i,k) for (i,k) in members) ./ length(members)
        c = Makie.to_color(c)
        csalpha = [Makie.RGBAf(c.r, c.g, c.b, 0.0), Makie.RGBAf(c.r, c.g, c.b, 1.0)]
        colorrange = something(_group_colorrange(equilibrium_densities, species, model, members), Makie.automatic)
        plt = Makie.heatmap!(ax, X, Y, Z; colormap=csalpha, colorrange=colorrange, label=_maybe_texlabel(label,latex; upright=true))
    end

    if x_units == :normalized
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]).*1e9)
        x_norm = "nm"
    else
        Makie.xlims!(ax,(bounds[1][1],bounds[1][2]))
        x_norm = "m"
    end

    if y_units == :normalized
        Makie.ylims!(ax,(bounds[2][1],bounds[2][2])./L)
        y_norm = "σ"
    elseif y_units == :angstrom
        Makie.ylims!(ax,(bounds[2][1],bounds[2][2]).*1e10)
        y_norm = "Å"
    elseif y_units == :nanometer
        Makie.ylims!(ax,(bounds[2][1],bounds[2][2]).*1e9)
        y_norm = "nm"
    else
        Makie.ylims!(ax,(bounds[2][1],bounds[2][2]))
        y_norm = "m"
    end

    ax.xlabel = _maybe_texlabel("x / "*x_norm,latex)
    ax.ylabel = _maybe_texlabel("y / "*y_norm,latex)

    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.plot(system::Union{ClassicalDFT.DFTSystem,ClassicalDFT.DGTSystem,ClassicalDFT.SCFTSystem}, structure::ClassicalDFT.DFTStructure{3,ClassicalDFT.Cartesian,M}, profiles; x_units=:normalized, y_units=:normalized, latex=false, plot_by=:bead, color_by=:bead, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false, equilibrium_densities=nothing, only=nothing) where M
    _check_profile_color_by(plot_by, color_by)
    structure = system.structure
    model = system.model
    species = system.species
    _ρ = Adapt.adapt(CPU(), profiles)

    x = ClassicalDFT.uniform_range(structure,1)
    y = ClassicalDFT.uniform_range(structure,2)
    z = ClassicalDFT.uniform_range(structure,3)
    L = ClassicalDFT.length_scale(model)

    if x_units == :normalized
        X, Y, Z = x./L, y./L, z./L
        x_norm = "σ"
    elseif x_units == :angstrom
        X, Y, Z = x.*1e10, y.*1e10, z.*1e10
        x_norm = "Å"
    elseif x_units == :nanometer
        X, Y, Z = x.*1e9, y.*1e9, z.*1e9
        x_norm = "nm"
    else
        X, Y, Z = x, y, z
        x_norm = "m"
    end

    fig = _cdft_figure(width, dpi, font)
    ax = Axis3(fig[1, 1];
        aspect=:data,
        xgridvisible=grid, ygridvisible=grid, zgridvisible=grid,
        xspinesvisible=false, yspinesvisible=false, zspinesvisible=false,
        xtickformat=_latex_tickformat(latex), ytickformat=_latex_tickformat(latex), ztickformat=_latex_tickformat(latex),
        xlabel=_maybe_texlabel("x / "*x_norm,latex),
        ylabel=_maybe_texlabel("y / "*x_norm,latex),
        zlabel=_maybe_texlabel("z / "*x_norm,latex))

    function bead_ρ(i, k)
        norm_const = _norm_const(species, model, i, k)
        return _ρ[:,:,:,k].*norm_const
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by, color_scheme)
    groups, colors = _filter_groups(groups, colors, only)

    plt = nothing
    for ((label, members), c) in zip(groups, colors)
        ρk = sum(bead_ρ(i,k) for (i,k) in members) ./ length(members)
        ρmin, ρmax = something(_group_colorrange(equilibrium_densities, species, model, members), extrema(ρk))
        # Clamped rather than left to float outside [0,1]: with a fixed `equilibrium_densities`
        # range (as opposed to this frame's own `extrema`), a transient profile can briefly
        # overshoot the final bulk values (noisy initial guess, finite-size ringing, ...).
        normed = clamp.((ρk .- ρmin) ./ (ρmax - ρmin + 1e-8), 0, 1)

        c = Makie.to_color(c)
        cmap = [Makie.RGBAf(c.r, c.g, c.b, 0.45*a^2) for a in range(0,1;length=256)]

        plt = Makie.volume!(ax, extrema(X), extrema(Y), extrema(Z), normed;
            algorithm=:absorption, absorption=5f0, colormap=cmap)
    end

    return Makie.FigureAxisPlot(fig, ax, plt)
end

end
