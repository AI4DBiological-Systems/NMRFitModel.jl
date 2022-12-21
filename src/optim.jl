"""
```
runOptimjl(x_initial::Vector{T},
    f::Function,
    df!::Function, g_tol::T;
    x_tol::T = zero(T),
    f_tol::T = zero(T),
    max_time::T = convert(T, Inf),
    max_iters::Int = 100000,
    lp::Int = 2,
    verbose::Bool = false,
) where T <: AbstractFloat
```
"""
function runOptimjl(
    x_initial::Vector{T},
    f::Function,
    df!::Function,
    g_tol::T;
    x_tol::T = zero(T),
    f_tol::T = zero(T),
    max_time::Real = Inf,
    max_iters::Integer = 100000,
    lp::Integer = 2,
    show_trace = false,
    verbose::Bool = false,
    ) where T <: AbstractFloat

    @assert g_tol > zero(T)

    optim_config = Optim.Options(
        x_tol = x_tol,
        f_tol = f_tol,
        g_tol = g_tol,
        iterations = max_iters,
        time_limit = max_time,
        show_trace = show_trace,
        extended_trace = show_trace, # show extended version of trace if we are showing trace at all.
    )

    ret = Optim.optimize(f, df!, x_initial, 
        Optim.ConjugateGradient(
            #linesearch = Optim.LineSearches.HagerZhang(),
            #linesearch = Optim.LineSearches.BackTracking(),
            #linesearch = Optim.LineSearches.Static(),
            

            #alphaguess = Optim.LineSearches.InitialHagerZhang(),
            #alphaguess = Optim.LineSearches.InitialQuadratic(),
        ), optim_config)

    x_star::Vector{T} = convert(Vector{T}, Optim.minimizer(ret))

    df_x_star = similar(x_initial)
    df!(df_x_star, x_star)
    df_norm::T = convert(T, norm(df_x_star, lp))

    if verbose
        @show ret
        println()
    end

    return x_star, df_norm, ret.ls_success, ret.iterations
end

