
# order of points in X matters.
# based on https://sites.cs.ucsb.edu/~veronika/MAE/summary_SLINK_Sibson72.pdf
function singlelinkage(
    X::Vector{Vector{T}};
    early_stop_distance = Inf, # default to solve for all possible levels.
    ) where T <: AbstractFloat

    # set up.
    h_set = Vector{T}(undef,0) # fusion distance at each level.
    partition_set = Vector{Vector{Vector{Int}}}(undef,0)
    
    # leaf level.
    rs, cs, ds = setupdistancebuffers(X) 
    ts = tuple.(rs, cs)# each index corresponds to a pair of parts in the partition for the current level.

    push!(h_set, zero(T))
    push!(partition_set, collect([i;] for i in eachindex(X)))
    
    partition_prev = partition_set[end]

    # subsequent levels.
    new_part_ind = length(X) # we only merge two parts at each level, so only one new partition index per level.

    # each iteration creates a level.
    # continue if
    #   -previous level's fusion distance has not triggered the treshold yet,
    #   - and if the previous level yielded a non-singleton partition.
    while length(ds) > 0 && h_set[end] < early_stop_distance

        #@show ts, ds, partition_prev
        #display([1:length(ts) ts ds])

        @assert length(ts) == length(ds)
        @assert length(partition_set) == length(h_set)

        partition = deepcopy(partition_prev)

        min_val, min_ind = findmin(ds)
        # r = rs[min_ind]
        # c = cs[min_ind]
        a1, a2 = ts[min_ind]
        #@show a1, a2, min_ind

        new_part_ind = mergeparts!(partition, a1, a2)
        push!(partition_set, partition)
        push!(h_set, min_val)

        updatedistancebuffers!(ts, ds, a1, a2, new_part_ind)

        # prepare for next round.
        partition_prev = partition

        #println()
    end

    return h_set, partition_set
end

# i, j are the point index for which parts to merge.
# Do nothing if they are already in the same partition. Otherwise, merge.
# If merging: removes the parts that contain points i, j. 
# If merging: add the union of the two parts to the last entry of `partition`.
function mergeparts!(partition::Vector{Vector{Int}}, i::Int, j::Int)::Int

    #
    inds = findall(xx->(any(xx .== i) || any(xx .== j)), partition)

    if length(inds) != 2
        #println("Warning: mergepartition() found more than two parts with the specify point indices. Skip merging.")
        return 0
    end

    part_for_merging1 = partition[inds[begin]]
    part_for_merging2 = partition[inds[begin+1]]

    deleteat!(partition, inds)

    new_part_ind = length(partition)
    push!(partition, [part_for_merging1; part_for_merging2])

    return new_part_ind
end

# convention used here is that r > c, rs[i] > cs[i] for any valid i.
# i, j are the point index for which parts to merge.
# Do nothing if they are already in the same partition. Otherwise, merge.
# If merging: removes the parts that contain points i, j. 
# If merging: add the union of the two parts to the last entry of `partition`.
function updatedistancebuffers!(
    ts::Vector{Tuple{Int,Int}},
    ds::Vector{T},
    a1::Int,
    a2::Int,
    new_part_ind::Int,
    ) where T

    @assert length(ts) == length(ds)
    
    affected_pairs = getaffectedpairs(ts, a1, a2)

    del_list = Vector{Int}(undef, 0)
    new_ts = Vector{Tuple{Int,Int}}(undef, 0)
    new_ds = Vector{T}(undef, 0)

    processed_ks = Vector{Int}(undef, 0) # use this to avoid double processing.

    #new_part_ind = length(ts) + 1

    for n in eachindex(affected_pairs)

        pair_ind = affected_pairs[n]
        t = ts[pair_ind]

        if t == (a1,a2) || t == (a2,a1)
            
            # this pair does not survive in sequent levels.
            push!(del_list, pair_ind)

            #println("if")
            #@show del_list, t
        else
        
            i, k, j = enforceorder(t, a1, a2)

            if typeof(findfirst(xx->xx==k, processed_ks)) <: Nothing

                inds = findpairs(ts, i, k)
                if length(inds) != 1
                    println("Error: problem with distance buffer; cannot find unique entry for (c,k) = ($i,$k).")
                end
                k_i = inds[begin]

                inds = findpairs(ts, j, k)
                if length(inds) != 1
                    println("Error: problem with distance buffer; cannot find unique entry for (c,k) = ($j,$k).")
                end
                k_j = inds[begin]
                
                # add one new pair.
                new_d = min(ds[k_i], ds[k_j])
                push!(new_ds, new_d)

                #new_part_ind += 1
                push!(new_ts, (new_part_ind, k))

                # discard the two pairs.
                push!(del_list, k_i)
                push!(del_list, k_j)

                # book keep to avoid double processing.
                push!(processed_ks, k)

                # println("else")
                # @show del_list, t
                # @show (new_part_ind, k)
                # @show i, k, j
            end
            #@show i, k, j
        end
    end

    unique!(del_list)
    sort!(del_list)
    deleteat!(ts, del_list)
    deleteat!(ds, del_list)

    push!(ts, new_ts...)
    push!(ds, new_ds...)
  
    return nothing
end
# # test code.
# Random.seed!(25)
# X = collect( randn(3) for _ = 1:4 )
# rs, cs, ds = setupdistancebuffers(X) 
# ts = tuple.(rs,cs)

# display([1:length(ts) ts ds])
# updatedistancebuffers!(ts, ds, 2, 3, 11)
# display([1:length(ts) ts ds])



function getdistance(part::Vector{Int}, ds::Vector{T}, k::Int)::T where T
    
    min_distance = convert(T, Inf)
    for n in eachindex(part)
        
        i = part[n]
        if min_distance > ds[i]
            min_distance = ds[i]
        end
    end
    
    return min_distance
end

function setupdistancebuffers(X::Vector{Vector{T}}) where T
    
    N = length(X)
    M = div(N*(N-1),2)

    rs = Vector{Int}(undef, M)
    cs = Vector{Int}(undef, M)
    ds = Vector{T}(undef, M)
    
    i = 0
    for c in eachindex(X)
        for r in eachindex(X)[begin:c-1]

            i += 1
            rs[i] = r
            cs[i] = c
            ds[i] = norm(X[r]-X[c])
        end
    end

    return rs, cs, ds
end


function findpairs(ts::Vector{Tuple{Int,Int}}, a1::Int, a2::Int)
    out = findall(xx->( xx==(a1,a2) || xx==(a2,a1) ), ts)

    return out
end

function hascomponent(x, a::Int)
    return any(x .== a)
end

function getaffectedpairs(ts::Vector{Tuple{Int,Int}}, a1::Int, a2::Int)
    return findall(xx->( hascomponent(xx, a1) || hascomponent(xx, a2) ), ts)
end


function enforceorder(t::Tuple{Int,Int}, a::Int)
    if t[begin] == a
        return a, t[end]
    elseif t[end] == a
        return a, t[begin]
    end

    return 0, 0 # indicate a is not found in t.
end

function enforceorder(t::Tuple{Int,Int}, a1::Int, a2::Int)

    i, k = enforceorder(t, a1)
    j = a2
    if i == 0
        i, k = enforceorder(t, a2)
        j = a1
    end

    return i, k, j
end

##### applications.

function mergepoints(X::Vector{Vector{T}}; tol = 1e-6) where T

    @assert tol > zero(T)

    h_set, partition_set = singlelinkage(X; early_stop_distance = tol)

    inds = sortperm(h_set, rev = true)

    partitioned_set_sorted = partition_set[inds]
    h_set_sorted = h_set[inds]

    ind = findfirst(xx->xx<tol, h_set_sorted)
    if typeof(ind) <: Nothing
        println("Error with mergepoints(). Returning a copy of the input.")
        return copy(X)
    end

    P = partitioned_set_sorted[ind] # select partition.

    K = length(P) # number of parts.
    out = Vector{Vector{T}}(undef, K)
    for k in eachindex(out)
        out[k] = Statistics.mean(X[P[k]])
    end

    return out
end