
#### generic.

function initializeparameter(αs::Vector{Vector{T}}, default_val::T) where T <: AbstractFloat
    out = similar(αs)
    
    for n in eachindex(αs)
        out[n] = Vector{T}(undef, length(αs[n]))
        fill!(out[n], default_val)
    end

    return out
end

### parsing.
function exportvalues!(
    x::Vector{T},
    p::CLSingletParameters{T};
    ind = 1,
    ) where T

    N_singlets = getNsinglets(p.αs)
    N_vars = 3*N_singlets

    x[begin+ind-1:begin+ind+N_vars-2] = [
        collect( Iterators.flatten(p.τs) );
        collect( Iterators.flatten(p.βs) );
        collect( Iterators.flatten(p.ξs) );
    ]

    return ind+N_vars-1 # last index that was updated.
end


############ model specific

function parsevalues!(
    p::CLSingletParameters{T},
    x::Vector{T};
    st_ind = 1,
    ) where T

    N_singlets = getNsinglets(p.αs)

    fin_ind = updateparameters!(p.τs, x; ind = st_ind)
    @assert fin_ind == st_ind + N_singlets - 1
    st_ind = fin_ind +1

    fin_ind = updateparameters!(p.βs, x; ind = st_ind)
    @assert fin_ind == st_ind + N_singlets - 1
    st_ind = fin_ind +1
    
    fin_ind = updateparameters!(p.ξs, x; ind = st_ind)
    @assert fin_ind == st_ind + N_singlets - 1
    st_ind = fin_ind +1

    return nothing
end

# for a parameter set of a mixture of singlets; i.e., xs is either τs, βs, or ξs.
function updateparameters!(
    xs::Vector{Vector{T}},
    p;
    ind::Int = 1) where T <: AbstractFloat

    for n in eachindex(xs)
        for i in eachindex(xs[n])

            xs[n][i] = p[begin+ind-1]
            ind += 1
        end
    end

    return ind - 1 # last index that was used to update.
end

