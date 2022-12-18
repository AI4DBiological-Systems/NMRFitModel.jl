
# run load_setup.jl first.

include("./helpers/utils.jl")
include("./helpers/verify.jl")


using SparseArrays

setupwsolver = NMRFitModel.setupwsolver

Random.seed!(25)

# set up.
model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = w)
lbs, ubs = NMRSignalSimulator.fetchbounds(model_params, Bs; shift_proportion = 0.9)



# constants.
shifts, phases, T2s = MSS.shifts, MSS.phases, MSS.T2s
mapping = NMRSignalSimulator.getParamsMapping(shifts, phases, T2s)

U = U_cost
#y = y_cost
y = randn(Complex{Float64}, length(y_cost))

# variable.
p_test = generateparameters(lbs, ubs)

# under test.
X, gs_re, gs_im, shift_multi_inds, phase_multi_inds,
    T2_multi_inds = NMRSignalSimulator.costfuncsetup(mapping, MSS, U)
C = NMRSignalSimulator.CostFuncBuffer(
    X, gs_re, gs_im, shift_multi_inds, phase_multi_inds, T2_multi_inds)

grad_p = ones(Float64, NMRSignalSimulator.getNvars(model_params))
costfunc! = pp->NMRSignalSimulator.evalcost!(
    grad_p,
    model_params,
    pp,
    C,
    y,
)
c = costfunc!(p_test)

# # oracle.
# costfunc0 = pp->NMRSignalSimulator.evalcost(model_params, pp, y, U)
# c_oracle = costfunc0(p_test)

# #@show c_oracle, c
# @show norm(c_oracle-c)






# # Envelope theorem.

# set up constraints.
lb = 0.0
ub = 1e3 # constraints not binding at optimal.

BLS_params = setupwsolver(X, MSS, lb, ub, y)


# make sure the cost runs.
fill!(BLS_params.primal_initial, 0.0)
envcostfunc = pp->NMRFitModel.evalenvelopecost!(
    model_params,
    BLS_params,
    C, 
    pp,
    y,
)
env_cost = envcostfunc(p_test)

@show env_cost

# make sure analytical derivative runs.
fill!(grad_p, NaN)
dEC_AN! = (gg,pp)->NMRFitModel.evalenvelopegradient!(
    gg,
    model_params,
    BLS_params,
    C,
    pp,
    y,
)
dEC_AN!(grad_p, p_test)
grad_env_AN = grad_p

# numerical gradient.
ND_accuracy_order = 8
dEC_ND = pp->FiniteDifferences.grad(
    FiniteDifferences.central_fdm(
        ND_accuracy_order,
        1,
    ),
    envcostfunc,
    pp
)[1]

println("Timing: dEC_ND")
@time grad_env_ND = dEC_ND(p_test)

@show norm(grad_env_AN)
@show norm(grad_env_ND)

println("Absolute discrepancy:")
@show norm(grad_env_ND - grad_env_AN)
println()

println("Relative discrepancy:")
@show norm(grad_env_ND - grad_env_AN)/norm(grad_env_AN)
println()

