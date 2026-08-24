module PlotsClassicalDFTExt

using ClassicalDFT
using Plots
import Plots: Colors

_maybe_texlabel(s, latex::Bool) = latex ? ClassicalDFT.texlabel(s) : s

# Figure-level style kwargs shared by every `Plots.plot` method below, ported 1:1 from the
# house rcParams style (see ClassicalDFT.CLASSICALDFT_* constants in src/utils/plot_style.jl): pixel size
# (from `width ∈ (:single,:double)` at `dpi`), white background, tick/label/legend font
# sizes, font family (`font` overrides the latex-conditional default when given), a
# categorical color cycle (`color_scheme`, default ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS), and grid
# on/off (`grid`, default `false` -- unchanged current look, but now overridable; when
# `true`, uses ClassicalDFT.CLASSICALDFT_GRID_COLOR/CLASSICALDFT_GRID_LINESTYLE).
function _cdft_base_plot(latex::Bool, font, width::Symbol, dpi::Real, grid::Bool, color_scheme)
    return Plots.plot(
        size = ClassicalDFT.classicaldft_figure_size(width, dpi),
        dpi = dpi,
        background_color = :white,
        color_palette = Plots.palette(color_scheme),
        grid = grid ? :on : :off,
        gridcolor = ClassicalDFT.CLASSICALDFT_GRID_COLOR,
        gridstyle = grid ? Symbol(ClassicalDFT.CLASSICALDFT_GRID_LINESTYLE == "-" ? :solid : :dash) : :solid,
        framestyle=:box,
        foreground_color_legend = nothing,
        xtickfontsize=ClassicalDFT.CLASSICALDFT_TICK_LABELSIZE,
        ytickfontsize=ClassicalDFT.CLASSICALDFT_TICK_LABELSIZE,
        xlabelfontsize=ClassicalDFT.CLASSICALDFT_AXES_LABELSIZE,
        ylabelfontsize=ClassicalDFT.CLASSICALDFT_AXES_LABELSIZE,
        # NOTE: must call `Plots.font` fully-qualified here, not the bare `font` that
        # `using Plots` would normally bring into scope -- this function's own `font`
        # parameter (the font-family override, a String or `nothing`) shadows it.
        legend_font=Plots.font(ClassicalDFT.CLASSICALDFT_LEGEND_FONTSIZE),
        fontfamily = font !== nothing ? font : (latex ? "Computer Modern" : :default),
    )
end

function Plots.plot(system::ClassicalDFT.AbstractClassicalDFTSystem, profiles; x_units=:normalized, y_units=:normalized, latex=false, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false)
    return Plots.plot(system, system.structure, profiles; x_units=x_units, y_units=y_units, latex=latex, color_scheme=color_scheme, font=font, width=width, dpi=dpi, grid=grid)
end

function Plots.plot(system::ClassicalDFT.AbstractClassicalDFTSystem, structure::ClassicalDFT.DFTStructure1DCart, profiles; x_units=:normalized, y_units=:mass, latex=false, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false)
    structure = system.structure
    model = system.model
    if model isa ClassicalDFT.ElectrolyteModel
        model = model.neutralmodel
    end
    species = system.species
    nb = length(profiles)

    bounds = structure.bounds

    z = ClassicalDFT.uniform_range(structure, 1)
    L = ClassicalDFT.length_scale(model)

    plt = _cdft_base_plot(latex, font, width, dpi, grid, color_scheme)

    ymax = 0.
    species_id = 1
    bead_id = 1
    for i in ClassicalDFT.@comps
        for k in ClassicalDFT.@chain(i)
            if species.nbeads[i] > 1
                species_name = model.components[i]
                group_name = model.groups.flattenedgroups[k]
                name = "$species_name $group_name"
                norm_const = model.params.segment[k]*species.size[k]^3*ClassicalDFT.N_A
            else
                species_name = model.components[i]
                name = "$species_name"
                norm_const = model.params.segment[i]*species.size[i]^3*ClassicalDFT.N_A
            end

            if x_units == :normalized
                X = z./L
            elseif x_units == :angstrom
                X = z.*1e10
            elseif x_units == :nanometer
                X = z.*1e9
            else
                X = z
            end

            if y_units == :normalized
                Y = profiles[:,k].*norm_const
                y_norm = "σ³"
            elseif y_units == :mass
                Mw = model.params.Mw[k]
                Y = profiles[:,k].*Mw/1e3
                y_norm = " / (kg/m³)"
            elseif y_units == :angstrom
                Y = profiles[:,k].*ClassicalDFT.N_A/1e30
                y_norm = " / (kg/m³)"
            else
                Y = profiles[:,k]
                y_norm = " / (mol/m³)"
            end

            Plots.plot!(plt,X,Y,label=_maybe_texlabel(name,latex),linewidth=3)
            ymax = max(ymax,maximum(Y))
        end
    end


    if x_units == :normalized
        Plots.xlims!(plt,(bounds[1],bounds[2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Plots.xlims!(plt,(bounds[1],bounds[2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Plots.xlims!(plt,(bounds[1],bounds[2]).*1e9)
        x_norm = "nm"
    else
        Plots.xlims!(plt,(bounds[1],bounds[2]))
        x_norm = "m"
    end

    Plots.ylims!(plt,(0,1.1*ymax))
    Plots.xlabel!(plt,_maybe_texlabel("z / "*x_norm,latex))

    if y_units == :normalized
        Plots.ylabel!(plt,_maybe_texlabel("ρσ³",latex))
    elseif y_units == :mass
        Plots.ylabel!(plt,_maybe_texlabel("ρ / (kg/m³)",latex))
    else
        Plots.ylabel!(plt,_maybe_texlabel("ρ / (mol/m³)",latex))
    end

    Plots.plot!(plt,legend=:topleft)

    return plt
end

function Plots.plot(system::ClassicalDFT.AbstractClassicalDFTSystem, structure::Union{ClassicalDFT.DFTStructure1DSphr,ClassicalDFT.DFTStructure1DCyl}, profiles; x_units=:normalized, y_units=:mass, latex=false, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false)
    structure = system.structure
    model = system.model
    if model isa ClassicalDFT.ElectrolyteModel
        model = model.neutralmodel
    end
    species = system.species
    nb = length(profiles)

    bounds = structure.bounds

    z = ClassicalDFT.structure_r(structure)
    L = ClassicalDFT.length_scale(model)

    plt = _cdft_base_plot(latex, font, width, dpi, grid, color_scheme)

    ymax = 0.
    for i in ClassicalDFT.@comps
        for k in ClassicalDFT.@chain(i)
            if species.nbeads[i] > 1
                species_name = model.components[i]
                group_name = model.groups.flattenedgroups[k]
                name = "$species_name $group_name"
                norm_const = model.params.segment[k]*species.size[k]^3*ClassicalDFT.N_A
            else
                species_name = model.components[i]
                name = "$species_name"
                norm_const = model.params.segment[i]*species.size[i]^3*ClassicalDFT.N_A
            end

            if x_units == :normalized
                X = z./L
            elseif x_units == :angstrom
                X = z.*1e10
            elseif x_units == :nanometer
                X = z.*1e9
            else
                X = z
            end

            if y_units == :normalized
                Y = profiles[:,k].*norm_const
                y_norm = "σ³"
            elseif y_units == :mass
                Mw = model.params.Mw[k]
                Y = profiles[:,k].*Mw/1e3
                y_norm = " / (kg/m³)"
            elseif y_units == :angstrom
                Y = profiles[:,k].*ClassicalDFT.N_A/1e30
                y_norm = " / (kg/m³)"
            else
                Y = profiles[:,k]
                y_norm = " / (mol/m³)"
            end

            Plots.plot!(plt,X,Y,label=_maybe_texlabel(name,latex),linewidth=3)
            ymax = max(ymax,maximum(Y))
        end
    end

    if x_units == :normalized
        Plots.xlims!(plt,(bounds[1],bounds[2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Plots.xlims!(plt,(bounds[1],bounds[2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Plots.xlims!(plt,(bounds[1],bounds[2]).*1e9)
        x_norm = "nm"
    else
        Plots.xlims!(plt,(bounds[1],bounds[2]))
        x_norm = "m"
    end

    Plots.ylims!(plt,(0,1.1*ymax))
    Plots.xlabel!(plt,_maybe_texlabel("r / "*x_norm,latex))

    if y_units == :normalized
        Plots.ylabel!(plt,_maybe_texlabel("ρσ³",latex))
    elseif y_units == :mass
        Plots.ylabel!(plt,_maybe_texlabel("ρ / (kg/m³)",latex))
    else
        Plots.ylabel!(plt,_maybe_texlabel("ρ / (mol/m³)",latex))
    end

    Plots.plot!(plt,legend=:topleft)

    return plt
end

function Plots.plot(system::Union{ClassicalDFT.DFTSystem,ClassicalDFT.DGTSystem}, structure::ClassicalDFT.DFTStructure2DCart, profiles; x_units=:normalized, y_units=:normalized, latex=false, color_scheme=ClassicalDFT.CLASSICALDFT_DEFAULT_COLORS, font=nothing, width=:single, dpi=ClassicalDFT.CLASSICALDFT_DPI, grid=false)
    # Per-species base colors for the heatmap alpha-gradients below (a *categorical*
    # color-per-species assignment, distinct from `color_palette` used for line plots
    # elsewhere in this file) -- was hardcoded to `palette(:tab10)`, now driven by the same
    # `color_scheme` kwarg as every other method, cycling via `mod1` the same way
    # MakieClassicalDFTExt's `_assign_colors` does.
    colors = Plots.palette(color_scheme)
    structure = system.structure
    model = system.model
    species = system.species
    nb = length(profiles)

    bounds = structure.bounds

    x = ClassicalDFT.uniform_range(structure,1)
    y = ClassicalDFT.uniform_range(structure,2)
    X = zeros(length(x),length(y))
    Y = zeros(length(x),length(y))

    for i in 1:length(x)
        X[i,:] .= x[i]
    end

    for i in 1:length(y)
        Y[:,i] .= y[i]
    end

    L = ClassicalDFT.length_scale(model)

    plt = _cdft_base_plot(latex, font, width, dpi, grid, color_scheme)

    ymax = 0.
    species_id = 1
    bead_id = 1
    for i in ClassicalDFT.@comps
        for k in ClassicalDFT.@chain(i)
            if species.nbeads[i] > 1
                species_name = model.components[i]
                group_name = model.groups.flattenedgroups[k]
                name = "$species_name $group_name"
                norm_const = model.params.segment[k]*species.size[k]^3*ClassicalDFT.N_A
            else
                species_name = model.components[i]
                name = "$species_name"
                norm_const = model.params.segment[i]*species.size[i]^3*ClassicalDFT.N_A
            end

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
            elseif x_units == :angstrom
                Y = y.*1e10
            elseif x_units == :nanometer
                Y = y.*1e9
            else
                Y = y
            end

            Z = profiles[:,:,k].*norm_const
            z_norm = "σ³"

            # if y_units == :normalized
            #     Z = profiles[:,k].*norm_const
            #     y_norm = "σ³"
            # elseif y_units == :mass
            #     Mw = model.params.Mw[k]
            #     Y = profiles[:,k].*Mw/1e3
            #     y_norm = " / (kg/m³)"
            # elseif y_units == :angstrom
            #     Y = profiles[:,k].*ClassicalDFT.N_A/1e30
            #     y_norm = " / (kg/m³)"
            # else
            #     Y = profiles[:,k]
            #     y_norm = " / (mol/m³)"
            # end
            csalpha = cgrad([Colors.RGBA(colors[k].r, colors[k].g, colors[k].b, 0), Colors.RGBA(colors[k].r, colors[k].g, colors[k].b, 1)])
            Plots.heatmap!(plt,X,Y,Z,label=_maybe_texlabel(name,latex), c=csalpha)
        end
    end
    

    if x_units == :normalized
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2]).*1e9)
        x_norm = "nm"
    else
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2]))
        x_norm = "m"
    end

    if y_units == :normalized
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2])./L)
        y_norm = "σ"
    elseif y_units == :angstrom
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2]).*1e10)
        y_norm = "Å"
    elseif y_units == :nanometer
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2]).*1e9)
        y_norm = "nm"
    else
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2]))
        y_norm = "m"
    end

    Plots.xlabel!(plt,_maybe_texlabel("x / "*x_norm,latex))
    Plots.ylabel!(plt,_maybe_texlabel("y / "*y_norm,latex))

    # if y_units == :normalized
    #     Plots.ylabel!(plt,"ρσ³")
    # elseif y_units == :mass
    #     Plots.ylabel!(plt,"ρ / (kg/m³)")
    # else
    #     Plots.ylabel!(plt,"ρ / (mol/m³)")
    # end

    # Plots.plot!(plt,legend=:topleft)

    return plt
end

end