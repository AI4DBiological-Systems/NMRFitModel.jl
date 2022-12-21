
struct BLSMatrixType{T}
    A::Matrix{Complex{T}}
    t_range
    p::Vector{T} # model hyperparameters.
end

function constructdesignmatrix!(
    X::BLSMatrixType{T},
    evalmodelfunc,
    )::Matrix{T} where T <: AbstractFloat

    A, t_range, p = X.A, X.t_range, X.p

    @assert size(A,1) == length(t_range)

    for n in axes(A,2)
        for m in axes(A,1)

            A[m,n] = evalmodelfunc(t_range[m], p, n)
        end
    end
    
    B = reinterpret(T, A)
    return B
end


function evalcostfunc(
    p::Vector{T2},
    w::Vector{T},
    y::Vector{Complex{T}},
    t_range,
    modelfunc0,
    ) where {T,T2}

    cost = zero(T)
    for m in eachindex(t_range)
        cost += abs2( modelfunc0(t_range[m], p, w) - y[m] )
    end

    return cost
end

# every interation, solve for BLS.
function evalenvelopecost!(
    mat_params,
    BLS_params, # BLS inputs. BLS solves for the basis coefficients.
    f, # basis funcs.

    p::Vector{T},
    y::Vector{Complex{T}},
    t_range,
    modelfunc0, # sum of basis funcs.
    ) where T

    # BLS.
    primal_sol, dual_sol, status_flag, obj_val = NMRFitModel.solveBLS!(
        BLS_params,
        constructdesignmatrix!(mat_params, f),
    )
    #w = primal_sol
    w = collect(
        clamp(
            primal_sol[i],
            BLS_params.lbs[i],
            BLS_params.ubs[i]
        ) for i in eachindex(BLS_params.ubs)
    )
    BLS_params.primal_initial[:] = w

    # cost.
    return evalcostfunc(p, w, y, t_range, modelfunc0)
end

function evalenvelopegradient!(
    mat_params,
    BLS_params, # BLS inputs. BLS solves for the basis coefficients.
    f, # basis funcs.

    p::Vector{T},
    y::Vector{Complex{T}},
    t_range,
    modelfunc0, # sum of basis funcs.
    ) where T

    # BLS.
    primal_sol, dual_sol, status_flag, obj_val = NMRFitModel.solveBLS!(
        BLS_params,
        constructdesignmatrix!(mat_params, f),
    )
    #w = primal_sol
    w = collect(
        clamp(
            primal_sol[i],
            BLS_params.lbs[i],
            BLS_params.ubs[i]
        ) for i in eachindex(BLS_params.ubs)
    )
    BLS_params.primal_initial[:] = w

    # cost.
    modelfunc = pp->evalcostfunc(pp, w, y, t_range, modelfunc0)

    return ForwardDiff.gradient(modelfunc, p)
end