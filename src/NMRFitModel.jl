module NMRFitModel


import OSQP
using SparseArrays
using LinearAlgebra

#include("types.jl")
#include("parse.jl")

include("BLS.jl")

#include("complex_lorentzian/mixture_model.jl")
#include("complex_lorentzian/models.jl")




end # module NMRFitModel
