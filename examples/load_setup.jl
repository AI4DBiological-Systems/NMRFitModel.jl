
Random.seed!(25)

include("./helpers/data.jl")
include("./helpers/utils.jl")
include("./helpers/SH.jl")

PyPlot.close("all")
fig_num = 1
PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])




# # Load model and data
#project_name = "NRC-4_amino_acid-Jan2022-1-D2O"
#project_name = "NRC-4_amino_acid-Jan2022-1-DSS"
#project_name = "Serine-700MHz"
project_name = "NRC-Glucose-2018"

load_folder_path = joinpath("./output", project_name)


dict_Phys = JSON3.read(
    read(
        joinpath(load_folder_path, "Phys.json"),
    ),
)
Phys, molecule_entries = NMRHamiltonian.deserializephysicalparams(Dict(dict_Phys))



save_dir = "./output"
load_folder_path = joinpath(save_dir, project_name)

dict_As = JSON3.read(
    read(
        joinpath(load_folder_path, "As.json"),
    ),
)
As = NMRHamiltonian.deserializemixture(dict_As)


dict_Bs = JSON3.read(
    read(
        joinpath(load_folder_path, "Bs.json"),
    ),
)
ss_params_set, op_range_set, λ0 = deserializclproxies(dict_Bs)

dict_itp = BSON.load(
    joinpath(load_folder_path, "itp.bson"),
)
itp_samps2 = deserializitpsamples(dict_itp)

Bs, MSS = NMRSignalSimulator.recoverclproxies(
    itp_samps2,
    ss_params_set,
    op_range_set,
    As,
    λ0,
)

experiment_dict = BSON.load(
    joinpath(load_folder_path, "experiment.bson"),
)
s_t =experiment_dict[:s_t]
fs = experiment_dict[:fs]
SW = experiment_dict[:SW]
ν_0ppm = experiment_dict[:ν_0ppm]
α_0ppm = experiment_dict[:α_0ppm]
β_0ppm = experiment_dict[:β_0ppm]
λ_0ppm = experiment_dict[:λ_0ppm]

hz2ppmfunc = uu->(uu - ν_0ppm)*SW/fs
ppm2hzfunc = pp->(ν_0ppm + pp*fs/SW)

# # Specify cost intervals.
# ## user inputs.
#molecule_entries = ["alpha-D-Glucose"; "beta-D-Glucose"; "D2O"; ]

w = ones(length(molecule_entries))
# end user inputs.


# # Specify region for fit.

u_offset = 0.2 #in units ppm.
Δcs_padding = 0.02 #in units ppm.
min_window_cs = 0.06 #in units ppm.


## frequency locations. For plotting.
ΩS_ppm = getPsnospininfo(As, hz2ppmfunc)
ΩS_ppm_sorted = sort(combinevectors(ΩS_ppm))


u_min = ppm2hzfunc(ΩS_ppm_sorted[1] - u_offset)
u_max = ppm2hzfunc(ΩS_ppm_sorted[end] + u_offset)

# This is the frequency range that we shall work with.
P = LinRange(hz2ppmfunc(u_min), hz2ppmfunc(u_max), 50000)
U = ppm2hzfunc.(P)
U_rad = U .* (2*π)

## get intervals.
ΩS0 = getΩS(As)
ΩS0_ppm = getPs(ΩS0, hz2ppmfunc)

Δcs_padding = 0.1 # units are in ppm.
Δsys_cs = initializeΔsyscs(As, Δcs_padding)
exp_info = NMRSpecifyRegions.setupexperimentresults(molecule_entries, ΩS0_ppm, Δsys_cs; min_dist = Δcs_padding)



q = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Bs; w = w)

# evaluate at the plotting positions.
q_U = q.(U_rad)


band_inds, band_inds_set = NMRSpecifyRegions.getcostinds(exp_info, P)

U_cost = U[band_inds]
P_cost = P[band_inds]

y_cost = q_U[band_inds]


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P, real.(q_U), label = "simulated spectrum")
PyPlot.plot(P_cost, real.(y_cost), "x", label = "positions in band_inds")

PyPlot.legend()
PyPlot.xlabel("ppm")
PyPlot.ylabel("real")
PyPlot.title("positions against simulated spectrum, real part")

# ## translate this to the cost.


# visualize data spectrum.
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


# ## get cost from data spectrum.

# reduce magnitude of signal.
intensity_0ppm = maximum(abs.(q_U))
y = S_U ./ intensity_0ppm

# get cost.
band_inds, band_inds_set = NMRSpecifyRegions.getcostinds(exp_info, P_y)

U_cost = U_y[band_inds]
P_cost = P_y[band_inds]

y_cost = y[band_inds]


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P_y, real.(y), label = "scaled data spectrum")
PyPlot.plot(P_cost, real.(y_cost), "x", label = "positions in band_inds")

PyPlot.legend()
PyPlot.xlabel("ppm")
PyPlot.ylabel("real")
PyPlot.title("positions against scaled data spectrum, real part")
