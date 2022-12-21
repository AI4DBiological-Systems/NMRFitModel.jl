
# download the NMRData repository at https://github.com/AI4DBiological-Systems/NMRData

# include("../src/NMRDataSetup.jl")
# import .NMRDataSetup

import NMRDataSetup

#using FFTW
#import BSON

import Random
Random.seed!(25)

using DataDeps
import Tar

using FFTW

import PyCall
import PyPlot
import Conda
PyPlot.close("all")
fig_num = 1
PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

#import PythonCall # if not using PyCall nor PyPlot. On Linux, cannot use both PyCall/PyPlot with PythonCall as of Dec 2022.

include("./helpers/data.jl")
include("./helpers/utils.jl")
include("./helpers/SH.jl")

root_data_path = getdatapath() # coupling values data repository root path


### user inputs.
solvent_ppm_guess = 4.7
solvent_window_ppm = 0.1

#save_dir = "/outputs/NMR/experiments/NRC"
#project_name = "NRC-4_amino_acid-Jan2022-1"
##molecule_entries = ["alpha-D-Glucose"; "beta-D-Glucose"; "D2O"; ] # got w = [0; 0; w_ub;] # correct later.

# project_name = "NRC-4_amino_acid-Jan2022-1"
# relative_file_path = "experiments_1D1H/NRC/amino_acid_mixture_2022"

# molecule_entries = ["alpha-D-Glucose"; "beta-D-Glucose"; "DSS"; ]

# project_name = "NRC-Glucose-2018"
# relative_file_path = "experiments_1D1H/NRC/misc/glucose_2018"

# experiment_full_path = joinpath(root_data_path, relative_file_path)

### overside, serine 700 MHz.
experiment_full_path = "/home/roy/Documents/repo/NMRData/experiments_1D1H/BMRB/similar_settings/BMRB-700-20mM/L-Serine"

project_name = "Serine-700MHz"
molecule_entries = ["L-Serine"; ]

### end inputs.

## load.


# PyCall version.
# if nmrglue is not installed in the Python environment PyCall is associated with, PyCall uses Conda.jl to install it.
try
    global ng = PyCall.pyimport("nmrglue")
catch
    Conda.add("nmrglue"; channel="bioconda")
end

ng = PyCall.pyimport("nmrglue")
dic, data = ng.bruker.read(experiment_full_path) # PyCall auto-converts to Julia objects.
TD, SW, SFO1, O1, fs = extractexperimentparams(dic)
# end PyCall version.

# ## PyhonCall version.
# ng = PythonCall.pyimport("nmrglue")
# dic_py, data_py = ng.bruker.read(experiment_full_path)
# data = PythonCall.pyconvert(Vector{Complex{Float64}}, data_py)

# TD_py, SW_py, SFO1_py, O1_py, fs_py = extractexperimentparams(dic_py)
# TD = PythonCall.pyconvert(Int, TD_py)
# SW = PythonCall.pyconvert(Float64, SW_py)
# SFO1 = PythonCall.pyconvert(Float64, SFO1_py)
# O1 = PythonCall.pyconvert(Float64, O1_py)
# fs = PythonCall.pyconvert(Float64, fs_py)
# ## end PythonCall

s_t, S, hz2ppmfunc, ppm2hzfunc, ν_0ppm, α_0ppm, β_0ppm, λ_0ppm, Ω_0ppm,
α_solvent, β_solvent, λ_solvent, Ω_solvent, results_0ppm,
    results_solvent = NMRDataSetup.loadspectrum(data;
    N_real_numbers_FID_data = TD, # this should be equal to length(data)/2
    spectral_width_ppm = SW,
    carrier_frequency_Hz = SFO1,
    carrier_frequency_offset_Hz = O1,
    fs_Hz = fs,
    solvent_ppm = solvent_ppm_guess,
    solvent_window_ppm = solvent_window_ppm)

# ## store.
# isdir(save_dir) || mkdir(save_dir); # make save folder if it doesn't exist.
# save_folder_path = joinpath(save_dir, project_name)
# isdir(save_folder_path) || mkpath(save_folder_path); # make save folder if it doesn't exist.

# save_path = joinpath(save_folder_path, "experiment.bson")
# BSON.bson(save_path,
# s_t = s_t,
# fs = fs,
# SW = SW,
# ν_0ppm = ν_0ppm,
# α_0ppm = α_0ppm,
# β_0ppm = β_0ppm,
# λ_0ppm = λ_0ppm)

# ## Visualize.
hz2ppmfunc = uu->(uu - ν_0ppm)*SW/fs
ppm2hzfunc = pp->(ν_0ppm + pp*fs/SW)

offset_Hz = ν_0ppm - (ppm2hzfunc(0.3)-ppm2hzfunc(0.0))

N = length(s_t)
DFT_s = fft(s_t)
U_DFT, U_y, U_inds = NMRDataSetup.getwraparoundDFTfreqs(N, fs, offset_Hz)

# shift the DFT so that around 0 ppm (around `offset_Hz` Hz`) appears first in the frequency positions.
S_U = DFT_s[U_inds]
P_y = hz2ppmfunc.(U_y)

q = uu->NMRDataSetup.evalcomplexLorentzian(uu, α_0ppm, β_0ppm, λ_0ppm, 2*π*ν_0ppm)*fs
q_U = q.(U_y)


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P_y, real.(S_U), label = "data")
PyPlot.plot(P_y, real.(q_U), "--", label = "estimated lorentzian")

PyPlot.legend()
PyPlot.xlabel("ppm")
PyPlot.ylabel("real")
PyPlot.title("data")

####### simulate.

### inputs.


root_data_path = getdatapath() # coupling values data repository root path

H_params_path = joinpath(root_data_path, "coupling_info") # folder of coupling values. # replace with your own values in actual usage.

molecule_mapping_root_path = joinpath(root_data_path, "molecule_name_mapping")
molecule_mapping_file_path = joinpath(molecule_mapping_root_path, "select_molecules.json")

max_partition_size_offset = 2
λ0 = λ_0ppm
Δr_default = 1.0 # the samples used to build the surrogate is taken every `Δr` radian on the frequency axis. Decrease for improved accuracy at the expense of computation resources.
Δκ_λ_default = 0.05 # the samples used to build thes urrogate for κ_λ are taken at this sampling spacing. Decrease for improved accuracy at the expense of computation resources.
Δcs_max_scalar_default = 0.08 # In units of ppm. interpolation border that is added to the lowest and highest resonance frequency component of the mixture being simulated.
κ_λ_lb_default = 0.5 # interpolation lower limit for κ_λ.
κ_λ_ub_default = 2.5 # interpolation upper limit for κ_λ.

#type_SSParams = NMRSignalSimulator.getSpinSysParamsdatatype(NMRSignalSimulator.SharedShift{Float64})
type_SSParams = NMRSignalSimulator.getSpinSysParamsdatatype(NMRSignalSimulator.CoherenceShift{Float64})


### end inputs.



Phys = NMRHamiltonian.getphysicalparameters(
    Float64,
    molecule_entries,
    H_params_path,
    molecule_mapping_file_path;
    unique_cs_atol = 1e-6,
)

As, Rs, Phys = runSH(
    Phys,
    fs,
    SW,
    ν_0ppm,
    molecule_entries,
    max_partition_size_offset;
    #search_θ = true,
    #θ_default = 0.0,
    starting_manual_knn = 60,
    γ_base = 0.8,
    #γ_rate = 1.05,
    max_iters_γ = 100,
    #min_dynamic_range = 0.95,
    cc_gap_tol = 1e-8,
    cc_max_iters = 300,
    assignment_zero_tol = 1e-3,
)


Bs, MSS, itp_samps = NMRSignalSimulator.fitclproxies(type_SSParams, As, λ0;
    names = molecule_entries,
    #config_path = surrogate_config_path,
    Δcs_max_scalar_default = Δcs_max_scalar_default,
    κ_λ_lb_default = κ_λ_lb_default,
    κ_λ_ub_default = κ_λ_ub_default,
    # u_min = u_min,
    # u_max = u_max,
    Δr_default = Δr_default,
    Δκ_λ_default = Δκ_λ_default,
)

w = ones(length(molecule_entries))

## save.
save_dir = "./output"
save_folder_path = joinpath(save_dir, project_name)
isdir(save_folder_path) || mkpath(save_folder_path); # make save folder if it doesn't exist.


BSON.bson(
    joinpath(save_folder_path, "experiment.bson"),
    s_t = s_t,
    fs = fs,
    SW = SW,
    ν_0ppm = ν_0ppm,
    α_0ppm = α_0ppm,
    β_0ppm = β_0ppm,
    λ_0ppm = λ_0ppm,
)


S_Phys = NMRHamiltonian.serializephysicalparams(Phys, molecule_entries)
NMRHamiltonian.saveasJSON(
    joinpath(save_folder_path, "Phys.json"),
    S_Phys,
)


S_As = NMRHamiltonian.serializemixture(As)
NMRHamiltonian.saveasJSON(
    joinpath(save_folder_path, "As.json"),
    S_As,
)


S_Bs = serializclproxies(Bs)
NMRHamiltonian.saveasJSON(
    joinpath(save_folder_path, "Bs.json"),
    S_Bs,
)

# ## itp samps.
S_itp = serializitpsamples(itp_samps)
BSON.bson(
    joinpath(save_folder_path, "itp.bson"),
    S_itp, # only support BSON loading for this.
)

