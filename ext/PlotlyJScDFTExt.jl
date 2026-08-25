module PlotlyJScDFTExt
using ClassicalDFT
using PlotlyJS

function PlotlyJS.plot(system::ClassicalDFT.DFTSystem, profiles; x_units=:normalized, y_units=:normalized)
    return PlotlyJS.plot(system, system.structure, profiles; x_units=x_units, y_units=y_units)
end

function PlotlyJS.plot(system::ClassicalDFT.DFTSystem, structure::ClassicalDFT.DFTStructure1DCart, profiles; x_units = :normalized, y_units = :normalized)

    model = system.model
    species = system.species
    nb = length(profiles)

    bounds = structure.bounds

    z = ClassicalDFT.uniform_range(structure, 1)
    L = ClassicalDFT.length_scale(model)

    colors = ["rgb(31, 119, 180, 1)",
              "rgb(255, 127, 14, 1)",
              "rgb(44, 160, 44, 1)",
              "rgb(214, 39, 40, 1)",
              "rgb(148, 103, 189, 1)",
              "rgb(140, 86, 75, 1)",
              "rgb(227, 119, 194, 1)",
              "rgb(127, 127, 127, 1)",
              "rgb(188, 189, 34, 1)",
              "rgb(23, 190, 207, 1)"]


    layout = PlotlyJS.Layout(autosize=false,width=700,height=470,
             xaxis = PlotlyJS.attr(title = "Temperature  / K", font_size=12, showgrid=false,            
                                    ticks="inside",mirror=true,showline=true,linecolor="black"),
             yaxis = PlotlyJS.attr(title = "Density / (mol/dm³)", font_size=12, showgrid=false,       
                                    ticks="inside",mirror=true,showline=true,linecolor="black"),
             showlegend=false, plot_bgcolor="white")

    trace = PlotlyJS.GenericTrace[]

    ymax = 0.
    for i in ClassicalDFT.@comps
        for k in ClassicalDFT.@chain(i)

            if species.nbeads[i] > 1
                species_name = model.components[i]
                group_name = model.groups.flattenedgroups[k]
                name = "$species_name $group_name"
                norm_const = model.params.segment[k]*species.size[k]^3*ClassicalDFT.N_A
                level = (1.5-((system.species.levels[k]-1)/maximum(system.species.levels)))/1.5
                # find the ith color in the color scheme 
                color = colors[i]
                # reduce saturation based on level
                r, g, b = parse.(Int, split(color[5:end-1], ", "))
                r = round(Int, 1/(1/r*level))
                g = round(Int, 1/(1/g*level))
                b = round(Int, 1/(1/b*level))
                color = "rgb($r, $g, $b)"
            else
                species_name = model.components[i]
                name = "$species_name"
                norm_const = model.params.segment[i]*species.size[i]^3*ClassicalDFT.N_A

                color = colors[i]
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
        

            append!(trace, [PlotlyJS.scatter(;x=X, y=Y, mode="lines", name=name, line_color=color, line=attr(width=2))])
            ymax = max(ymax,maximum(Y))
        end
    end

    if x_units == :normalized
        layout[:xaxis][:range] = [bounds[1], bounds[2]]./L
        layout[:xaxis][:title] = "z / σ"
    elseif x_units == :angstrom
        layout[:xaxis][:range] = [bounds[1], bounds[2]].*1e10
        layout[:xaxis][:title] = "z / Å"
    elseif x_units == :nanometer
        layout[:xaxis][:range] = [bounds[1], bounds[2]].*1e9
        layout[:xaxis][:title] = "z / nm"
    else
        layout[:xaxis][:range] = [bounds[1], bounds[2]]
        layout[:xaxis][:title] = "z / m"
    end

    if y_units == :normalized
        layout[:yaxis][:range] = [0, 1.1*ymax]
        layout[:yaxis][:title] = "ρσ³"
    elseif y_units == :mass
        layout[:yaxis][:range] = [0, 1.1*ymax]
        layout[:yaxis][:title] = "ρ / (kg/m³)"
    else 
        layout[:yaxis][:range] = [0, 1.1*ymax]
        layout[:yaxis][:title] = "ρ / (mol/m³)"
    end
    
    plt = PlotlyJS.plot(trace, layout)

    return plt
end

function PlotlyJS.plot(system::ClassicalDFT.DFTSystem, structure::ClassicalDFT.DFTStructure2DCart, profiles; x_units = :normalized, y_units = :normalized)

    colors = ["rgba(31, 119, 180, 1)",
              "rgba(255, 127, 14, 1)",
              "rgba(44, 160, 44, 1)",
              "rgba(214, 39, 40, 1)",
              "rgba(148, 103, 189, 1)",
              "rgba(140, 86, 75, 1)",
              "rgba(227, 119, 194, 1)",
              "rgba(127, 127, 127, 1)",
              "rgba(188, 189, 34, 1)",
              "rgba(23, 190, 207, 1)"]

    model = system.model
    species = system.species
    nb = length(profiles)
    bounds = structure.bounds

    x = ClassicalDFT.uniform_range(structure,1)
    y = ClassicalDFT.uniform_range(structure,2)

    L = ClassicalDFT.length_scale(model)

    layout = PlotlyJS.Layout(autosize=false,width=700,height=470,
             xaxis = PlotlyJS.attr(font_size=12, showgrid=false,            
                                    ticks="inside",mirror=true,showline=true,linecolor="black"),
             yaxis = PlotlyJS.attr(font_size=12, showgrid=false,       
                                    ticks="inside",mirror=true,showline=true,linecolor="black"),
             showlegend=false, plot_bgcolor="white")

    trace = PlotlyJS.GenericTrace[]

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
                X = vec(x)./L
            elseif x_units == :angstrom
                X = vec(x).*1e10
            elseif x_units == :nanometer
                X = vec(x).*1e9
            else
                X = vec(x)
            end

            if y_units == :normalized
                Y = vec(y)./L
            elseif x_units == :angstrom
                Y = vec(y).*1e10
            elseif x_units == :nanometer
                Y = vec(y).*1e9
            else
                Y = vec(y)
            end

            Z = profiles[:,:,k].*norm_const

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
            # csalpha = cgrad([Colors.RGBA(colors[k].r, colors[k].g, colors[k].b, 0), Colors.RGBA(colors[k].r, colors[k].g, colors[k].b, 1)])
            
            colorscale = [[0, "rgba(255, 2555, 255,0)"], [1, colors[mod(i,10)]]]
            append!(trace, [PlotlyJS.heatmap(x=X, y=Y, z=Z', colorscale=colorscale, name=name, connectgaps=true, zsmooth="best", showscale=false)])
        end
    end

    if x_units == :normalized
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]]./L
        layout[:xaxis][:title] = "x / σ"
    elseif x_units == :angstrom
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]].*1e10
        layout[:xaxis][:title] = "x / Å"
    elseif x_units == :nanometer
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]].*1e9
        layout[:xaxis][:title] = "x / nm"
    else
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]]
        layout[:xaxis][:title] = "x / m"
    end

    if y_units == :normalized
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]]./L
        layout[:yaxis][:title] = "y / σ"
    elseif y_units == :angstrom
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]].*1e10
        layout[:yaxis][:title] = "y / Å"
    elseif y_units == :nanometer
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]].*1e9
        layout[:yaxis][:title] = "y / nm"
    else
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]]
        layout[:yaxis][:title] = "y / m"
    end
    
    plt = PlotlyJS.plot(trace, layout)

    return plt
end

function PlotlyJS.plot(system::ClassicalDFT.DFTSystem, structure::ClassicalDFT.DFTStructure3DCart, profiles; x_units = :normalized, y_units = :normalized)

    colors = ["rgba(31, 119, 180, 1)",
              "rgba(255, 127, 14, 1)",
              "rgba(44, 160, 44, 1)",
              "rgba(214, 39, 40, 1)",
              "rgba(148, 103, 189, 1)",
              "rgba(140, 86, 75, 1)",
              "rgba(227, 119, 194, 1)",
              "rgba(127, 127, 127, 1)",
              "rgba(188, 189, 34, 1)",
              "rgba(23, 190, 207, 1)"]

    model = system.model
    species = system.species
    nb = length(profiles)
    bounds = structure.bounds
    ngrid = structure.ngrid

    ρb = deepcopy(structure.ρbulk)

    _x = ClassicalDFT.uniform_range(structure,1)
    x = zeros(ngrid...)
    for i in 1:length(_x)
        x[i,:,:] .= _x[i]
    end
    _y = ClassicalDFT.uniform_range(structure,2)
    y = zeros(ngrid...)
    for i in 1:length(_y)
        y[:,i,:] .= _y[i]
    end
    _z = ClassicalDFT.uniform_range(structure,3)
    z = zeros(ngrid...)
    for i in 1:length(_z)
        z[:,:,i] .= _z[i]
    end

    L = ClassicalDFT.length_scale(model)

    layout = PlotlyJS.Layout(autosize=false,width=700,height=470,
             xaxis = PlotlyJS.attr(font_size=12, showgrid=false,            
                                    ticks="inside",mirror=true,showline=true,linecolor="black"),
             yaxis = PlotlyJS.attr(font_size=12, showgrid=false,       
                                    ticks="inside",mirror=true,showline=true,linecolor="black"),
             zaxis = PlotlyJS.attr(font_size=12, showgrid=false,       
                                    ticks="inside",mirror=true,showline=true,linecolor="black"),
             showlegend=false, plot_bgcolor="white")

    plt = PlotlyJS.plot()

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

            if y_units == :normalized
                Z = z./L
            elseif x_units == :angstrom
                Z = z.*1e10
            elseif x_units == :nanometer
                Z = z.*1e9
            else
                Z = z
            end

            W = profiles[:,:,:,k].*norm_const
            ρb[i] = ρb[i].*norm_const
            
            colorscale = [[0, "rgba(255,255,255,0)"], [1, colors[mod(i,10)]]]
            PlotlyJS.add_trace!(plt, PlotlyJS.volume(
                x=X[:],
                y=Y[:],
                z=Z[:],
                value=W[:],
                isomin=ρb[i]*0.85,
                isomax=maximum(W),
                colorscale=colorscale,
                showscale=false,
                surface_count=17, # needs to be a large number for good volume rendering
            ))
            # plt = PlotlyJS.plot()
        end
    end

    if x_units == :normalized
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]]./L
        layout[:xaxis][:title] = "x / σ"
    elseif x_units == :angstrom
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]].*1e10
        layout[:xaxis][:title] = "x / Å"
    elseif x_units == :nanometer
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]].*1e9
        layout[:xaxis][:title] = "x / nm"
    else
        layout[:xaxis][:range] = [bounds[1,1], bounds[1,2]]
        layout[:xaxis][:title] = "x / m"
    end

    if y_units == :normalized
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]]./L
        layout[:yaxis][:title] = "y / σ"
    elseif y_units == :angstrom
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]].*1e10
        layout[:yaxis][:title] = "y / Å"
    elseif y_units == :nanometer
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]].*1e9
        layout[:yaxis][:title] = "y / nm"
    else
        layout[:yaxis][:range] = [bounds[2,1], bounds[2,2]]
        layout[:yaxis][:title] = "y / m"
    end

    if y_units == :normalized
        layout[:zaxis][:range] = [bounds[3,1], bounds[3,2]]./L
        layout[:zaxis][:title] = "z / σ"
    elseif y_units == :angstrom
        layout[:zaxis][:range] = [bounds[3,1], bounds[3,2]].*1e10
        layout[:zaxis][:title] = "z / Å"
    elseif y_units == :nanometer
        layout[:zaxis][:range] = [bounds[3,1], bounds[3,2]].*1e9
        layout[:zaxis][:title] = "z / nm"
    else
        layout[:zaxis][:range] = [bounds[3,1], bounds[3,2]]
        layout[:zaxis][:title] = "z / m"
    end
    
    # plt = PlotlyJS.plot(trace[1])

    return plt
end

end