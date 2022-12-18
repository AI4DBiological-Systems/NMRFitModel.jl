


###### resonance plots


function setupplotintervals(exp_info; min_window_cs = -0.1)

    plot_lbs = collect( exp_info.regions[i].st for i in eachindex(exp_info.regions) )
    plot_ubs = collect( exp_info.regions[i].fin for i in eachindex(exp_info.regions) )
    sort!(plot_lbs)
    sort!(plot_ubs)

    if min_window_cs < 0
        return plot_lbs, plot_ubs
    end

    #A = plot_ubs-plot_lbs
    #Z = sum(A)
    #avg_length = 1/length(plot_lbs)

    for i in eachindex(plot_lbs)
        interval = plot_ubs[i] - plot_lbs[i]
        if interval < min_window_cs
            padding = min_window_cs - interval
            plot_ubs[i] += padding
            plot_lbs[i] -= padding
        end

        #if A[i]/Z < 0.8*avg_length # auto rescale interval might have convergence issue. opt for manual.
    end

    return plot_lbs, plot_ubs
end

# only use the first molecule entry.
function setupresonanceplotfuncs(As, Bs)::Tuple{Function,Vector{Vector{Function}},Function}
    
    A = As[1]
    B = Bs[1]
    
    # summed components.
    q = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Bs)

    # resonance groups for the non-singlet spin systems.
    qs = Vector{Vector{Function}}(undef, length(B.qs))
    for i in eachindex(B.qs)

        qs[i] = Vector{Function}(undef, length(B.qs[i]))
        for k in eachindex(B.qs[i])
            qs[i][k] = ωω->B.qs[i][k](ωω-B.ss_params.shift.ζ[i], B.ss_params.T2.var[i])
        end
    end

    # singlets.
    #q_singlets = ωω->NMRSignalSimulator.evalclsinglets(ωω, B.ζ_singlets, A.αs_singlets, A.Ωs_singlets, B.β_singlets, B.λ0, B.κs_λ_singlets)

    return q, qs#, q_singlets  
end

function setupresonanceplot(
    q::Function,
    qs::Vector{Vector{Function}},
    q_singlets::Function,
    ppm2hzfunc::Function;
    show_sum = true,
    #show_singlets = true,
    q_text = "sum",
    qs_prefix_text = "",
    q_singlets_text = "singlets",
    postfunc::Function = real,
    line_style_q = :dashdot,
    line_style_qs = :solid,
    line_style_q_singlets = :solid,
    sum_RGB_tuple = (0.0, 0.0, 0.0),
    singlets_RGB_tuple = (),
    ) where T <: Real

    ys = Vector{Function}(undef, 0)
    label_texts = Vector{String}(undef, 0)
    line_styles = Vector{Symbol}(undef, 0)

    if show_sum
        push!(ys, pp->postfunc( q(ppm2hzfunc(pp)*2*π) ))
        push!(label_texts, q_text)
        push!(line_styles, line_style_q)
    end

    for i in eachindex(qs)
        for k in eachindex(qs[i])
            push!(ys, pp->postfunc( qs[i][k](ppm2hzfunc(pp)*2*π) ))
            push!(label_texts, "$(qs_prefix_text)($(i),$(k))")
            push!(line_styles, line_style_qs)
        end
    end

    # if show_singlets
    #     push!(ys, pp->postfunc( q_singlets(ppm2hzfunc(pp)*2*π) ))
    #     push!(label_texts, q_singlets_text)
    #     push!(line_styles, line_style_q_singlets)
    # end

    # get colours, but force sum and singlets to the (if valid) specified colours
    seed_colours = [(1.0, 1.0, 1.0);
        (0.0, 0.0, 0.0)]
    if length(sum_RGB_tuple) == 3
        push!(seed_colours, sum_RGB_tuple)
    end
    if length(singlets_RGB_tuple) == 3
        push!(seed_colours, singlets_RGB_tuple)
    end
    seed_colours = unique(seed_colours) # up to floating-point euqality.
    #@show seed_colours

    plot_colours = MakiePlots.getplotcolors(
        length(ys),
        Float64;
        seeds = seed_colours,
        drop_seeds = true, # do not return seed colours.
    )
    if show_sum && length(sum_RGB_tuple) == 3
        plot_colours[1] = sum_RGB_tuple
    end

    if show_singlets && length(singlets_RGB_tuple) == 3
        plot_colours[end] = singlets_RGB_tuple
    end

    return ys, label_texts, line_styles, plot_colours
end



# resonance group simulations.
function setupresonanceplot0(qs_U::Vector{Vector{Vector{Complex{T}}}};
    group_prefix_text = "",
    processing_func::Function = real,
    q_U::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0),
    line_style_q_U = "--",
    line_style_qs_U = "-",
    #q_singlets_U::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0),
    #line_style_q_singlets_U = "-",
    ) where T <: Real

    ys = Vector{Vector{T}}(undef, 0)
    label_texts = Vector{String}(undef, 0)
    line_styles = Vector{String}(undef, 0)

    if !isempty(q_U)
        push!(ys, processing_func.(q_U))
        push!(label_texts, "sum")
        push!(line_styles, line_style_q_U)
    end

    for i in eachindex(qs_U)
        for k in eachindex(qs_U[i])
            push!(ys, processing_func.(qs_U[i][k]))
            push!(label_texts, "$(group_prefix_text) ($(i),$(k))")
            push!(line_styles, line_style_qs_U)
        end
    end

    # if !isempty(q_singlets_U)
    #     push!(ys, processing_func.(q_singlets_U))
    #     push!(label_texts, "singlets")
    #     push!(line_styles, line_style_q_singlets_U)
    # end

    return ys, label_texts, line_styles
end

