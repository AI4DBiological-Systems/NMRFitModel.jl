
# run load_setup.jl first. It'll choose the correct `project_name`.

import Statistics
import BSON

Random.seed!(25)

include("./helpers/data.jl")
include("./helpers/utils.jl")
include("./helpers/SH.jl")

include("./helpers/warping.jl")
include("./helpers/single_linkage.jl")

setupwsolver = NMRFitModel.setupwsolver
UseGradientTrait = NMRSignalSimulator.UseGradientTrait
IgnoreGradientTrait = NMRSignalSimulator.IgnoreGradientTrait

PyPlot.close("all")

#### load saved results.

#project_name = "NRC-4_amino_acid-Jan2022-1-D2O"
#project_name = "NRC-4_amino_acid-Jan2022-1-DSS"

dic = BSON.load("./output/$(project_name)/fit_results.bson")
x_initials = dic[:x_initials]
xs_star = dic[:xs_star]
f_star = dic[:f_star]

x_star_PO = dic[:x_star_PO]
w_PO = dic[:w_PO]

#status = dic[:status]
#iters_ran = dic[:iters_ran]

f_initial = dic[:f_initial]
f_discrepancies = dic[:f_discrepancies]
df_discrepancies = dic[:df_discrepancies]
shift_proportion = dic[:shift_proportion]
w_ub = dic[:w_ub]

@show norm(f_discrepancies)
@show norm(df_discrepancies)

#### set up model and cost function.
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
@show f(x_star_PO)

NMRSignalSimulator.importmodel!(model_params, x_star_PO)
model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = w_PO)
#lbs, ubs = NMRSignalSimulator.fetchbounds(model_params, Bs; shift_proportion = shift_proportion)

f, df!, fdf!, BLS_params = setupfit(
    model_params,
    MSS,
    y_cost,
    U_cost;
    #w_lb = -0.1,
    w_lb = 0.0,
    w_ub = w_ub,
)

#### pick solution with lowest cost.

min_cost, min_ind = findmin(f_star)

x_star = xs_star[min_ind]

grad_x_star = ones(length(x_star)) .* NaN
cost_star = fdf!(grad_x_star, x_star)

#@assert abs(f_star[min_ind] - cost_star) < 1e-12
@show abs(f_star[min_ind] - cost_star)


#### visualize.
U_rad = U_y .* (2*π)

## parameters that affect qs.
# A.ζ, A.κs_λ, A.κs_β
# A.ζ_singlets, A.αs_singlets, A.Ωs_singlets, A.β_singlets, A.λ0, A.κs_λ_singlets
q = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Bs; w = w)

q_U = q.(U_rad)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P_y, real.(y), label = "data")
PyPlot.plot(P_cost, real.(y_cost), "x", label = "fit positions")
PyPlot.plot(P_y, real.(q_U), label = "fit solution")


PyPlot.legend()
PyPlot.xlabel("ppm")
PyPlot.ylabel("real")
PyPlot.title("data vs. fit, real part")
