# based on env.jl
# run load_setup.jl first.

include("./helpers/utils.jl")

setupwsolver = NMRFitModel.setupwsolver
UseGradientTrait = NMRSignalSimulator.UseGradientTrait
IgnoreGradientTrait = NMRSignalSimulator.IgnoreGradientTrait

Random.seed!(25)

# set up.
model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = w)
lbs, ubs = NMRSignalSimulator.fetchbounds(model_params, Bs; shift_proportion = 0.9)

# constants.
y = y_cost
#y = randn(Complex{Float64}, length(y_cost))
U = U_cost

shifts, phases, T2s = MSS.shifts, MSS.phases, MSS.T2s
mapping = NMRSignalSimulator.getParamsMapping(shifts, phases, T2s)

# data container.
X, gs_re, gs_im, shift_multi_inds, phase_multi_inds,
T2_multi_inds = NMRSignalSimulator.costfuncsetup(mapping, MSS, U)

C = NMRSignalSimulator.CostFuncBuffer(
    X, gs_re, gs_im, shift_multi_inds, phase_multi_inds, T2_multi_inds)

# BLS.
lb = 0.0
ub = 1e3 # constraints not binding at optimal.
#ub = 1.0


# B = NMRSignalSimulator.constructdesignmatrix!(
#     UseGradientTrait(),
#     X,
#     MSS,
# )

# @assert 1==2

BLS_params = setupwsolver(X, MSS, lb, ub, y)
fill!(BLS_params.primal_initial, 0.0)

@show BLS_params.primal_initial


#####
println("start f.")

f = pp->NMRFitModel.evalenvelopecost!(
    model_params,
    BLS_params,
    C, 
    pp,
    y,
)

df! = (gg,pp)->NMRFitModel.evalenvelopegradient!(
    gg,
    model_params,
    BLS_params,
    C,
    pp,
    y,
)

fdf! = (gg,pp)->NMRFitModel.evalenvelopegradient!(
    gg,
    model_params,
    BLS_params,
    C,
    pp,
    y,
)

### run optim.

# initial iterate set to all zeros.
p_test = generateparameters(lbs, ubs)
x_initial = zeros(length(p_test))

# initial iterate from file.
dic = BSON.load("./output/fit_results.bson")
xs_star = dic[:xs_star]
f_star = dic[:f_star]
f_initial = dic[:f_initial]
f_discrepancies = dic[:f_discrepancies]
df_discrepancies = dic[:df_discrepancies]


@assert 1==2

f_tol = 0.0
x_tol = 0.0
g_tol = 1e-8

x_star, df_x_star_norm, status, iters_ran = NMRFitModel.runOptimjl(
    x_initial,
    f,
    df!,
    g_tol;
    x_tol = x_tol,
    f_tol = f_tol,
    max_time = Inf,
    max_iters = 1000,
    lp = 2,
    verbose = true,
)

grad_x = ones(length(x_initial)) .* NaN
cost_x_star = fdf!(grad_x, x_star)

println("sanity checks for f(x_star) and df(x_star). Passed?")
@show abs(f(x_star) - cost_x_star) < 1e-12
@show abs(norm(grad_x) - df_x_star_norm) < 1e-12
println()

println("initial vs. final iterates:")
@show f(x_initial), cost_x_star