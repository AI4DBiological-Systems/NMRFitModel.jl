include("./helpers/BLS_toy_model.jl")
#include("./helpers/verify.jl")

Random.seed!(25)


verbose_flag = false

D = 3
t_range = LinRange(-3,3, 1000)
G = randn(D,D)
G = G'*G
u = randn(D)
f = (tt,pp,nn)->Complex(sinc(nn*dot(pp .* tt, G* (pp .* tt))), cos(nn*dot(u, pp .* tt )))

modelfunc0 = (tt,pp,ww)->sum( ww[n]*f(tt, pp, n) for n in eachindex(ww) )

N = 4
w_oracle = rand(N) .* 100
p_oracle = randn(D)
modelfunc_oracle = tt->modelfunc0(tt, p_oracle, w_oracle)

y = modelfunc_oracle.(t_range)

# # Cost.

# ## BLS set up.

#p_test = randn(D)
p_test = p_oracle + 0.01*randn(D) # hopefully cost isn't too large.

@assert length(w_oracle) == N

mat_params = BLSMatrixType(
    Matrix{Complex{Float64}}(undef, length(t_range), length(w_oracle)),
    t_range,
    copy(p_test),
)


# set up constraints.
#lb = 0.0
lb = -1e3
ub = 1e3 # constraints not binding at optimal.
lb_OSQP = ones(N) .* lb
ub_OSQP = ones(N) .* ub

B = constructdesignmatrix!(mat_params, f)
BLS_params = NMRFitModel.setupBLS(
    B,
    NMRFitModel.reinterpretcomplexvector(y),
    lb_OSQP,
    ub_OSQP;
    eps_abs = 1e-14,
    eps_rel = 1e-12,
    max_iter = 4000,
    verbose = false,
    alpha = 1.0,
)

# # Constrained least-squares.

B = constructdesignmatrix!(mat_params, f)
primal_sol, dual_sol, status_flag, obj_val = NMRFitModel.solveBLS!(
    BLS_params,
    B,
)
cost_x_BLS = obj_val



# ## cost function

costfunc = ww->evalcostfunc(copy(p_test), ww, y, t_range, modelfunc0)

cost_x = costfunc(primal_sol)

A = mat_params.A

# least squares to QP objective omits a constant term and a scaling factor. Account for this before comparing with the QP objective.
cost_x_processed = 0.5*cost_x -0.5*norm(BLS_params.observations, 2)^2

@show cost_x, cost_x_processed, cost_x_BLS

println("modified least squares objective vs QP objective. Should be practically zero.")
@show abs(cost_x_processed - cost_x_BLS)

println("modified QP objective vs least squares objective. Should be practically zero.")
cost_x_BLS_processed = 2*cost_x_BLS + norm(BLS_params.observations, 2)^2
@show abs(cost_x - cost_x_BLS_processed)


ND_accuracy_order = 8

dC_ND = pp->FiniteDifferences.grad(
    FiniteDifferences.central_fdm(
        ND_accuracy_order,
        1,
    ),
    costfunc,
    pp
)[1]

dC_ND_eval = dC_ND(primal_sol)
@show norm(dC_ND_eval)
println()

# back up.
w_prev = copy(primal_sol)

# # Envelope theorem.

# make sure the cost runs.
fill!(BLS_params.primal_initial, 0.0)
envcostfunc = pp->evalenvelopecost!(
    mat_params,
    BLS_params,
    f,
    pp,
    y,
    t_range,
    modelfunc0,
)
env_cost = envcostfunc(p_test)

@show env_cost
#@assert norm(env_cost - cost_x) < 1e-9
@show norm(env_cost - cost_x)

# make sure analytical derivative runs.
dEC_AN = pp->evalenvelopegradient!(
    mat_params,
    BLS_params,
    f,
    pp,
    y,
    t_range,
    modelfunc0,
)
grad_env_AN = dEC_AN(p_test)

# numerical gradient.
dEC_ND = pp->FiniteDifferences.grad(
    FiniteDifferences.central_fdm(
        ND_accuracy_order,
        1,
    ),
    envcostfunc,
    pp
)[1]

grad_env_ND = dEC_ND(p_test)

println("Absolute discrepancy:")
@show norm(grad_env_ND - grad_env_AN)
println()

println("Relative discrepancy:")
@show norm(grad_env_ND - grad_env_AN)/norm(grad_env_AN)
println()

