
######## multi-start 

import Sobol
import Optim
import IntervalMonoFuncs
import Interpolations

include("./helpers/data.jl")
include("./helpers/utils.jl")
include("./helpers/SH.jl")

include("./helpers/warping.jl")
include("./helpers/single_linkage.jl")

Random.seed!(25)

PyPlot.close("all")
fig_num = 1
PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])



# set up.
model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = w)
lbs, ubs = NMRSignalSimulator.fetchbounds(model_params, Bs; shift_proportion = 0.9)



# constants.
shifts, phases, T2s = MSS.shifts, MSS.phases, MSS.T2s
mapping = NMRSignalSimulator.getParamsMapping(shifts, phases, T2s)

### 

N_starts = 1000
#N_starts = 100
p0s = getinitialiterates(lbs, ubs, N_starts)

# ## warp.

# ### set up.
window = 0.05
N_itp_samples = 40 # 60 actually breaks interpolation?!
domain_proportion = 0.15
#domain_proportion = 0.9



itp_a, itp_b, minxs, runoptimfunc, p_range = setupitpab(
    window,
    N_itp_samples,
    domain_proportion,
)


# ### test.

#xs = collect( ps[i][begin] for i in eachindex(ps))
x0s = collect( p0s[i][begin] for i in eachindex(p0s))


#ys = collect( ps[i][begin+1] for i in eachindex(ps))
y0s = collect( p0s[i][begin+1] for i in eachindex(p0s))

T = Float64
ref_ind = div(length(x0s),2)
#st = 8
st = 2
lb = lbs[st]
ub = ubs[st]

lb + (ub-lb)*domain_proportion

#x0s = LinRange(lb,ub,length(x0s))

x_ref = x0s[ref_ind]

target = convertcompactdomain(x_ref, lb, ub, zero(T), one(T))
a = itp_a(target)
b = itp_b(target)

u0s = convertcompactdomain.(x0s, lb, ub, zero(T), one(T))
vs = IntervalMonoFuncs.evalcompositelogisticprobit.(
    u0s,
    a,
    b,
    zero(T),
    one(T),
)

ys = convertcompactdomain.(vs, zero(T), one(T), lb, ub)

# ys = IntervalMonoFuncs.evalinversecompositelogisticprobit.(
#     x0s,
#     a,
#     b,
#     lb,
#     ub,
# )

t = 1:length(x0s)

PyPlot.figure(fig_num)
fig_num += 1

#PyPlot.plot(x0s, ys)
PyPlot.plot(x0s, ys, "x")

PyPlot.legend()
PyPlot.xlabel("src space")
PyPlot.ylabel("dest space")
PyPlot.title("effect of warping on one dim")



#@assert 1==2


p = p0s[begin]



#Random.seed!(36)
#Random.seed!(24)
# TODO: use multi-start nelder-mead instead of particle swarm for a gradient-free deterministic global swearch strategy.



p2 = copy(p)
warpparameter!(
    p2,
    mapping.shift,
    lbs,
    ubs,
    itp_a,
    itp_b,
)

# display([1:length(p) p p2])
# println()

ps = deepcopy(p0s)
warpparameters!(ps, mapping.shift, lbs, ubs, itp_a, itp_b)


xs = collect( ps[i][begin] for i in eachindex(ps))
x0s = collect( p0s[i][begin] for i in eachindex(p0s))

ys = collect( ps[i][begin+1] for i in eachindex(ps))
y0s = collect( p0s[i][begin+1] for i in eachindex(p0s))

# no change; should be a y = x line.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x0s, xs, "x")

PyPlot.legend()
PyPlot.xlabel("src space")
PyPlot.ylabel("dest space")
PyPlot.title("effect of warping on the ref dim")



inds = sortperm(y0s)
x2_0 = y0s[inds]
x2 = ys[inds]

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x2_0, x2, "x")

PyPlot.legend()
PyPlot.xlabel("src space")
PyPlot.ylabel("dest space")
PyPlot.title("effect of warping on the 2nd dim")




PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x0s, y0s, "x", label = "original")
PyPlot.plot(xs, ys, "x", label = "warped")

PyPlot.legend()
PyPlot.xlabel("ref dim")
PyPlot.ylabel("follow dim")
PyPlot.title("effect of warping on the first two dims")


