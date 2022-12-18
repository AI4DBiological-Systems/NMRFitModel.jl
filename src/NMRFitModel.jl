module NMRFitModel

using SparseArrays
using LinearAlgebra

import NMRSignalSimulator
NMRHamiltonian = NMRSignalSimulator.NMRHamiltonian
JSON3 = NMRHamiltonian.JSON3

import NMRSpecifyRegions

import OSQP
import Optim

#include("types.jl")
#include("parse.jl")

include("BLS.jl")
include("optim.jl")

#include("complex_lorentzian/mixture_model.jl")
#include("complex_lorentzian/models.jl")




end # module NMRFitModel
