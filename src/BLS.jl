
#### utilities
function reinterpretcomplexvector(y::Vector{Complex{T}})::Vector{T} where T <: AbstractFloat
    return reinterpret(T, y)
end

######### BLS


struct BLSParameters{T <: AbstractFloat}
    P_vec::Vector{T} # buffer.
    #A::Matrix{Complex{T}}
    optim_prob
    primal_initial::Vector{T}
    dual_initial::Vector{T}
    observations::Vector{T}
    lbs::Vector{T}
    ubs::Vector{T}
end


###### package up BLS. The update BLS thing doesn't seem to be working well.
# just re-setup every time we solve.

# no bound checking on v.
# upper triangule of A, including the diagonal.
function updateuppertriangular!(v, A)

    N = size(A,1)
    resize!(v, div(N*(N-1),2)+N) 

    k = 1
    for j in axes(A,2)
        for i in axes(A,1)[begin:begin+j-1]
            v[k] = A[i,j]
            k += 1
        end
    end

    return nothing
end

# https://github.com/osqp/OSQP.jl/issues/47
function setupBLS(
    #mat_params,
    B::Matrix{T},
    y::Vector{T}, # reinterpretcomplexvector(y)
    lbs::Vector{T},
    ubs::Vector{T};
    eps_abs = 1e-12,
    eps_rel = 1e-8,
    max_iter::Int = 4000,
    verbose::Bool = false,
    alpha = 1.0,
    adaptive_rho = false,
    show_initial = false
    ) where T <: AbstractFloat

    # set up model objects.
    #B::Matrix{T} = constructdesignmatrix!(mat_params)

    N = size(B,2)
    @assert length(lbs) == length(ubs) == N

    @assert length(y) == size(B,1)

    # turn them into quadratic program objects.
    P = sparse(B'*B)
    q = -B'*y
    G = sparse(LinearAlgebra.I, N, N)

    # buffer for P, for updating purpose.
    P_vec_buf = zeros(T, div(N*(N-1),2)+N) # upper triangule of P, including the diagonal.
    #updateuppertriangular!(P_vec_buf, P)

    # create the problem.
    prob = OSQP.Model()

    OSQP.setup!(
        prob;
        P = P, 
        q = q, 
        A = G, 
        l = lbs,
        u = ubs,
        alpha = alpha,
        eps_abs = eps_abs,
        eps_rel = eps_rel,
        max_iter = max_iter,
        verbose = verbose,
        adaptive_rho = adaptive_rho,
    )

    # assemble parameters and problem into a data structure.
    primal_initial = (lbs+ubs) ./2
    dual_initial = zeros(T,N)

    if show_initial
        @show primal_initial
        @show dual_initial
    end

    BLS_params = BLSParameters(
        P_vec_buf,
        #A_buf,
        prob,
        primal_initial,
        dual_initial,
        y,
        lbs,
        ubs,
    )

    return BLS_params
end

# OSQP docs: https://osqp.org/docs/interfaces/julia.html#solve
function solveBLS!(
    BLS_params::BLSParameters{T},
    #mat_params,
    B::Matrix{T},
    ) where T <: AbstractFloat

    prob, P_vec_buf, y = BLS_params.optim_prob, BLS_params.P_vec, BLS_params.observations
    primal_initial = BLS_params.primal_initial
    dual_initial = BLS_params.dual_initial

    ### debug.
    primal_initial = (BLS_params.lbs + BLS_params.ubs) ./ 2
    dual_initial = zeros(T, length(BLS_params.dual_initial))

    ### end debug.

    @assert length(primal_initial) == length(dual_initial)

    # update problem.
    #B = constructdesignmatrix!(mat_params)
    P = B'*B
    updateuppertriangular!(P_vec_buf, P)
    q_new = -B'*y

    OSQP.update!(
        prob,
        Px = P_vec_buf,
        q = q_new,
        l = BLS_params.lbs, # force update cache.
        u = BLS_params.ubs, # force update cache.
    )
    #@show norm(prob.ucache - BLS_params.ubs) if we don't force update the bounds, this is non-zero!
    #@show norm(prob.lcache - BLS_params.lbs)

    # solve.
    # @show primal_initial
    # @show dual_initial
    # @show P_vec_buf
    # @show q_new

    # force flush cache to ensure repeatable behavior.
    #fill!(prob.lcache,zero(T))
    #fill!(prob.ucache,zero(T))
    
    OSQP.warm_start!(prob; x = primal_initial, y = dual_initial)
    results = OSQP.solve!(prob)

    # prepare the results.
    primal_sol = results.x
    dual_sol = results.y

    status_flag = true
    if results.info.status != :Solved
        status_flag = false
    end

    obj_val = results.info.obj_val

    return primal_sol, dual_sol, status_flag, obj_val
end

########## for testing

# # test reinterpret on Matrix{Complex{T}}. it should be an interlaced matrix of double the number of rows.
# function interlacematrix(A::Matrix{Complex{T}})::Matrix{T} where T <: AbstractFloat

#     B = Matrix{T}(undef, size(A,1)*2, size(A,2))
#     for r in axes(A, 1)
#         for c in axes(A, 2)
#             B[2*(r-1)+1, c] = real(A[r,c])
#             B[2*(r-1)+2, c] = imag(A[r,c])
#         end
#     end

#     return B
# end



###### model-specific.

function setupwsolver(X, MSS, lb::T, ub::T, y::Vector{Complex{T}};
    eps_abs = 1e-14,
    eps_rel = 1e-12,
    max_iter = 4000,
    verbose = false,
    alpha = 1.0,
    ) where T

    N = size(X.A, 2)

    lb_OSQP = ones(N) .* lb
    ub_OSQP = ones(N) .* ub

    BLS_params = setupBLS(
        NMRSignalSimulator.constructdesignmatrix!(
            NMRSignalSimulator.UseGradientTrait(),
            X,
            MSS,
        ),
        reinterpretcomplexvector(y),
        lb_OSQP,
        ub_OSQP;
        eps_abs = eps_abs,
        eps_rel = eps_rel,
        max_iter = max_iter,
        verbose = verbose,
        alpha = alpha,
    )

    #fill!(BLS_params.primal_initial, 0.0)

    return BLS_params
end

function solvew!(
    MSS::NMRSignalSimulator.MixtureSpinSys,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    X::NMRSignalSimulator.BatchEvalBuffer{T},
    ) where T <: AbstractFloat

    primal_sol, dual_sol, status_flag, obj_val = solveBLS!(
        BLS_params,
        NMRSignalSimulator.constructdesignmatrix!(X, MSS),
    )
    w = collect(
        clamp(
            primal_sol[i],
            BLS_params.lbs[i],
            BLS_params.ubs[i]
        ) for i in eachindex(BLS_params.ubs)
    )
    #w = primal_sol # problematic. the residual floating point error causes line search assertion errors in Optim.jl.
    BLS_params.primal_initial[:] = w

    return w
end

# function updatew!(
#     model_params,
#     BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
#     X::NMRSignalSimulator.BatchEvalBuffer{T},
#     p::Vector{T},
#     ) where T <: AbstractFloat

#     MSS = model_params.MSS

#     NMRSignalSimulator.importmodel!(model_params, p)
#     NMRSignalSimulator.updatebuffer!(X, MSS)

#     w = solvew!(MSS, BLS_params, X)
#     model_params.w[:] = w

#     return nothing
# end

function updatewandderivatives!(
    model_params,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    X::NMRSignalSimulator.BatchEvalBuffer{T},
    p::Vector{T},
    ) where T <: AbstractFloat

    MSS = model_params.MSS

    NMRSignalSimulator.importmodel!(model_params, p)
    NMRSignalSimulator.updategradientbuffer!(X, MSS)

    w = solvew!(MSS, BLS_params, X)
    model_params.w[:] = w

    return nothing
end

# every interation, solve for BLS.
function evalenvelopecost!(
    model_params,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    C::NMRSignalSimulator.CostFuncBuffer{T},
    p::Vector{T},
    y::Vector{Complex{T}},
    ) where T <: AbstractFloat

    X = C.X
    updatewandderivatives!(model_params, BLS_params, X, p)
    w = model_params.w

    return norm(X.A *w - y)^2
end

function evalenvelopegradient!(
    grad_p::Vector{T},
    model_params,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    C::NMRSignalSimulator.CostFuncBuffer{T},
    p::Vector{T},
    y::Vector{Complex{T}},
    ) where T <: AbstractFloat

    updatewandderivatives!(model_params, BLS_params, C.X, p)

    # cost.
    cost = NMRSignalSimulator.evalcost!(
        grad_p,
        model_params,
        C,
        y,
    )

    return cost
end