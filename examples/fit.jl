# a.jl
# load_setup.jl
# multi-start.jl

include("./helpers/data.jl")
include("./helpers/utils.jl")
include("./helpers/SH.jl")

include("./helpers/warping.jl")
include("./helpers/single_linkage.jl")

Random.seed!(25)

PyPlot.close("all")

########### fit,
shift_proportion = 0.9
w_ub = 10.0
f_tol = 0.0
x_tol = 0.0
g_tol = 1e-8
max_iters = 1000
max_time = Inf

setupwsolver = NMRFitModel.setupwsolver
UseGradientTrait = NMRSignalSimulator.UseGradientTrait
IgnoreGradientTrait = NMRSignalSimulator.IgnoreGradientTrait

## for one start.
p_initials = mergepoints(ps; tol = 1e-6)


N = NMRSignalSimulator.getNentries(MSS)
w = ones(N)
model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = w)
lbs, ubs = NMRSignalSimulator.fetchbounds(model_params, Bs; shift_proportion = shift_proportion)

f, df!, fdf!, BLS_params = setupfit(
    model_params,
    MSS,
    y_cost,
    U_cost;
    #w_lb = -0.1,
    w_lb = 0.0,
    w_ub = w_ub,
)

## batch fit.
println("timing:")
#fill!(w, 1.0)
@time xs_star, f_star, f_initial, status, iters_ran,
    f_discrepancies, df_discrepancies = fitdata(
    w, # debug.
    BLS_params,
    model_params,
    f,
    df!,
    fdf!,
    #p_initials[1:10];
    p_initials;
    f_tol = f_tol,
    x_tol = x_tol,
    g_tol = g_tol,
    max_iters = max_iters,
    verbose = true,
    show_trace = false,
    max_time = max_time,
)
# 3569.479047

min_cost, min_ind = findmin(f_star)
x_star = xs_star[min_ind]
x_initial = p_initials[min_ind]

@show norm(f_discrepancies)
@show norm(df_discrepancies)
@show minimum(f_initial-f_star)

xs_star_cg = xs_star





## for one start, PO.

N = NMRSignalSimulator.getNentries(MSS)
w = ones(N)
model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = w)
lbs, ubs = NMRSignalSimulator.fetchbounds(model_params, Bs; shift_proportion = shift_proportion)

f, df!, fdf!, BLS_params, C = setupfit(
    model_params,
    MSS,
    y_cost,
    U_cost;
    #w_lb = -0.1,
    w_lb = 0.0,
    w_ub = w_ub,
)

x_star_cg = copy(x_star)
x_initial_PO = copy(x_star_cg)
#x_initial_PO = copy(p_initials[begin])
println("Timing: PO:")
@time x_star_PO, status, iters_ran = runOptimjlParticleSwarm(
    x_initial_PO,
    f,
    lbs,
    ubs;
    x_tol = x_tol,
    f_tol = f_tol,
    max_time = max_time,
    max_iters = 1000,
    #max_iters = 1,
    verbose = true,
)

println("PO solution, CG solution, zero solution")
@show f(x_star_PO), f(x_star_cg), norm(y)^2

# force run the solution iterate.
f(x_star_PO)
w_PO = copy(w)
#w_one = ones(length(As))

q = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Bs;
    #w = w_one,
    w = w,
)

U_rad = U_y .* (2*π)
q_U = q.(U_rad)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P_y, real.(y), label = "data")
PyPlot.plot(P_cost, real.(y_cost), "x", label = "fit positions")
PyPlot.plot(P_y, real.(q_U), label = "fit solution")


PyPlot.legend()
PyPlot.xlabel("ppm")
PyPlot.ylabel("real")
PyPlot.title("data vs. PO fit, real part")



import BSON

BSON.bson("./output/$(project_name)/fit_results.bson", 
    x_initials = p_initials,
    xs_star = xs_star,
    x_star_PO = x_star_PO,
    f_star = f_star,
    status = status,
    iters_ran = iters_ran,
    f_initial = f_initial, 
    f_discrepancies = f_discrepancies, 
    df_discrepancies = df_discrepancies,
    shift_proportion = shift_proportion,
    w_ub = w_ub,
    w_PO = w_PO,
)


# do distributed version. https://docs.julialang.org/en/v1/manual/distributed-computing/
