using BenchmarkTools
using Test

#import NMRHamiltonian
import NMRSpecifyRegions
import NMRDataSetup

using DataDeps
import Tar

import BSON

using LinearAlgebra
using FFTW

import Random
import Statistics

# # for plotting.
# import MakiePlots
# using Parameters

import PyPlot
import FiniteDifferences

using Revise
import NMRFitModel

#OSQP = NMRFitModel.OSQP

NMRSignalSimulator = NMRFitModel.NMRSignalSimulator
NMRHamiltonian = NMRSignalSimulator.NMRHamiltonian
JSON3 = NMRHamiltonian.JSON3

serializemixture = NMRHamiltonian.serializemixture
serializephysicalparams = NMRHamiltonian.serializephysicalparams

deserializemixture = NMRHamiltonian.deserializemixture
deserializephysicalparams = NMRHamiltonian.deserializephysicalparams

serializclproxies = NMRSignalSimulator.serializclproxies
serializitpsamples = NMRSignalSimulator.serializitpsamples

deserializclproxies = NMRSignalSimulator.deserializclproxies
deserializitpsamples = NMRSignalSimulator.deserializitpsamples

#include("load_setup.jl")
#include("multi-start.jl")
