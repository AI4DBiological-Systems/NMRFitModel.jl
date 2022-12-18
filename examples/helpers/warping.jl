######### warping.

"""
setupitpab(window::T, N_itp_samples::Int, domain_proportion::T;
   N_fit_positions::Int = 15,
   a_lb = 0.1,
   a_ub = 0.6,
   b_lb = -5.0,
   b_ub = 5.0,
   a_initial = 0.5,
   b_initial = 0.0,
   max_iters = 5000,
   xtol_rel = 1e-5,
   ftol_rel = 1e-5,
   maxtime = Inf,
   optim_algorithm = :LN_BOBYQA) where T <: Real
window ∈ (0,1)
optim_algorithm can be :GN_ESCH, :GN_ISRES, :LN_BOBYQA, :GN_DIRECT_L
"""
function setupitpab(window::T, N_itp_samples::Int, domain_proportion::T;
    N_fit_positions::Int = 15,
    a_lb = 0.1,
    a_ub = 0.6,
    b_lb = -5.0,
    b_ub = 5.0,
    a_initial = 0.5,
    b_initial = 0.0,
    max_iters = 5000,
    xtol_rel = 1e-5,
    ftol_rel = 1e-5,
    max_time = Inf,
    ) where T <: Real

    # get piece-wise linear monotone maps.
    infos, zs, p_range = IntervalMonoFuncs.createendopiewiselines1(
        zero(T), one(T),
    window; # try div 2 here next.
    N_itp_samples = N_itp_samples,
    domain_proportion = domain_proportion)

    # runoptimfunc() must return a 1D array of type Vector{T}, where T = eltype(pp0).
    # runoptimfunc = (pp0, ff, dff, pp_lb, pp_ub)->runoptimroutine(
    #     pp0, ff, dff, pp_lb, pp_ub;
    #     optim_algorithm = optim_algorithm,
    #     max_iters = max_iters,
    #     xtol_rel = xtol_rel,
    #     ftol_rel = ftol_rel,
    #     maxtime = maxtime)
    runoptimfunc = (pp0, ff, dff, pp_lb, pp_ub)->runOptimjlParticleSwarm(
        pp0, ff, pp_lb, pp_ub;
        max_iters = max_iters,
        x_tol = xtol_rel,
        f_tol = ftol_rel,
        max_time = max_time)

    # get compact sigmoid parameters fitted to each of the piece-wise linear maps.
    _, minxs = IntervalMonoFuncs.getlogisticprobitparameters(infos,
        runoptimfunc;
        N_fit_positions = N_fit_positions,
        a_lb = a_lb,
        a_ub = a_ub,
        b_lb = b_lb,
        b_ub = b_ub,
        a_initial = a_initial,
        b_initial = b_initial)


    Δp = p_range[2]-p_range[1]
    itp_range = p_range[1]:Δp:p_range[end]

    # itp.
    a_samples = collect( minxs[i][1] for i in eachindex(minxs) )
    b_samples = collect( minxs[i][2] for i in eachindex(minxs) )

    a_itp = Interpolations.interpolate(a_samples, Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid()))))
    a_sitp = Interpolations.scale(a_itp, itp_range)
    a_setp = Interpolations.extrapolate(a_sitp, Interpolations.Flat()) # take the closest boundary sample point as constant outside interp range.

    b_itp = Interpolations.interpolate(b_samples, Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid()))))
    b_sitp = Interpolations.scale(b_itp, itp_range)
    b_setp = Interpolations.extrapolate(b_sitp, Interpolations.Flat()) # take the closest boundary sample point as constant outside interp range.

    return a_setp, b_setp,
        minxs, runoptimfunc, p_range# debug
end

function runOptimjlParticleSwarm(
    x_initial::Vector{T},
    f::Function,
    lbs::Vector{T},
    ubs::Vector{T};
    x_tol::T = zero(T),
    f_tol::T = zero(T),
    max_time::Real = Inf,
    max_iters::Integer = 100000,
    n_particles::Integer = 10,
    verbose::Bool = false,
    ) where T <: AbstractFloat

    @assert length(x_initial) == length(lbs) == length(ubs)

    optim_config = Optim.Options(
        x_tol = x_tol,
        f_tol = f_tol,
        iterations = max_iters,
        time_limit = max_time,
    )
    swarm_config = Optim.ParticleSwarm(;
        lower = lbs,
        upper = ubs,
        n_particles = n_particles,
    )

    ret = Optim.optimize(f, x_initial, swarm_config, optim_config)

    #@show ret
    x_star::Vector{T} = convert(Vector{T}, Optim.minimizer(ret))

    if verbose
        @show ret
        println()
    end

    return x_star
end

function createendopiewiselines1(p_lb::T,
    p_ub::T,
    range_proportion::T;
    N_itp_samples::Int = 10,
    domain_proportion::T = 0.9) where T <: Real

    ## normalize input.
    #p_lb, p_ub, scale = normalizebounds(p_lb, p_ub)
    @assert -one(T) <= p_lb < p_ub <= one(T) # just don't allow other combinations for now.

    # set up.
    window = range_proportion*(p_ub-p_lb)/2

    #p_range = LinRange(p_lb + window + ϵ, p_ub - window - ϵ, N_itp_samples)
    p_range = LinRange(p_lb + window, p_ub - window, N_itp_samples)

    infos = Vector{IntervalMonoFuncs.Piecewise2DLineType{T}}(undef, length(p_range))
    zs = Vector{Vector{T}}(undef, length(p_range))

    for (i,p) in enumerate(p_range)

        intervals_y_st = [p - window;]
        intervals_y_fin = [p + window;]

        # returned scale is 1 and ignored here since we already normalized p_lb and p_ub to be in [-1,1].
        infos[i], _ = IntervalMonoFuncs.getpiecewiselines(
            intervals_y_st, intervals_y_fin, domain_proportion;
            lb = p_lb, ub = p_ub)

        zs[i] = [intervals_y_st[1]; intervals_y_fin[1]]
    end

    return infos, zs, p_range
end

# no error-checking.
function warpparameters!(
    ps::Vector{Vector{T}},
    mapping::NMRSignalSimulator.MoleculeParamsMapping,
    lbs::Vector{T},
    ubs::Vector{T},
    itp_a,
    itp_b,
    ) where T

    for p in ps
        warpparameter!(
            p::Vector{T},
            mapping::NMRSignalSimulator.MoleculeParamsMapping,
            lbs,
            ubs,
            itp_a,
            itp_b,
        )
    end

    return nothing
end
function warpparameter!(
    p::Vector{T},
    mapping::NMRSignalSimulator.MoleculeParamsMapping,
    lbs::Vector{T},
    ubs::Vector{T},
    itp_a,
    itp_b,
    ) where T

    for i in eachindex(mapping.st)
        
        st = mapping.st[i][begin]
        fin = mapping.fin[i][begin]

        if st != fin 
            # non-singlet case. do warping.
            
            # itp.
            x = p[st]
            lb = lbs[st]
            ub = ubs[st]

            target = convertcompactdomain(x, lb, ub, zero(T), one(T))
            #target = x
            a = itp_a(target)
            b = itp_b(target)

            for j = st+1:fin
                
                x2 = convertcompactdomain(p[j], lb, ub, zero(T), one(T))

                y = IntervalMonoFuncs.evalcompositelogisticprobit(
                    x2,
                    a,
                    b,
                    zero(T), #lb,
                    one(T), #ub,
                )

                #p[j] = y
                p[j] = convertcompactdomain(y, zero(T), one(T), lb, ub)

            end
        end
    end

    return nothing
end