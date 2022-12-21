
import Statistics

Random.seed!(25)

include("./helpers/data.jl")
include("./helpers/utils.jl")
include("./helpers/SH.jl")

include("./helpers/warping.jl")
include("./helpers/single_linkage.jl")

#### singlet-linkage clustering.

X = collect( randn(3) for _ = 1:4 )
rs, cs, ds = setupdistancebuffers(X) 
ts = tuple.(rs,cs)

#display([1:length(ts) ts ds])
# updatedistancebuffers!(ts, ds, 2, 3, 11)
# display([1:length(ts) ts ds])


h_set, partition_set = singlelinkage(X; early_stop_distance = 1.0)



Y = mergepoints(X; tol = 1.0)