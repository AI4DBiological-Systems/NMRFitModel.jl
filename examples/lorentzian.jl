# run a.jl
# import NMRHamiltonian
# import NMRSignalSimulator
# using DataDeps
# import Tar
# using LinearAlgebra
# import PyPlot

import Random
Random.seed!(25)

include("./helpers/data.jl")
include("./helpers/SH.jl")

PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

### user inputs.

#molecule_entries = ["L-Methionine"; "L-Phenylalanine"; "DSS"; "Ethanol"; "L-Isoleucine"]
#molecule_entries = ["alpha-D-Glucose"; "beta-D-Glucose"; "DSS"; "D2O"]
molecule_entries = ["alpha-D-Glucose"; "Ethanol"; "DSS"; "D2O"; "beta-D-Glucose"; ]

#molecule_entries = ["DSS"; ]
#molecule_entries = ["alpha-D-Glucose"; ]
#molecule_entries = ["D2O"; ]

root_data_path = getdatapath() # coupling values data repository root path

H_params_path = joinpath(root_data_path, "coupling_info") # folder of coupling values. # replace with your own values in actual usage.

molecule_mapping_root_path = joinpath(root_data_path, "molecule_name_mapping")
molecule_mapping_file_path = joinpath(molecule_mapping_root_path, "select_molecules.json")
#molecule_mapping_file_path = joinpath(molecule_mapping_root_path, "GISSMO_names.json")


# machine values taken from the BMRB 700 MHz 20 mM glucose experiment.
fs = 14005.602240896402
SW = 20.0041938620844
ν_0ppm = 10656.011933076665

w_oracle = rand(length(molecule_entries))
#w_oracle = ones(length(molecule_entries))

# # machine values for the BMRB 500 MHz glucose experiment.
# ν_0ppm = 6752.490995937095
# SW = 16.0196917451925
# fs = 9615.38461538462

max_partition_size_offset = 2

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

#@assert 1==2

λ0 = 3.4
Δr_default = 1.0 # the samples used to build the surrogate is taken every `Δr` radian on the frequency axis. Decrease for improved accuracy at the expense of computation resources.
Δκ_λ_default = 0.05 # the samples used to build thes urrogate for κ_λ are taken at this sampling spacing. Decrease for improved accuracy at the expense of computation resources.
Δcs_max_scalar_default = 0.2 # In units of ppm. interpolation border that is added to the lowest and highest resonance frequency component of the mixture being simulated.
κ_λ_lb_default = 0.5 # interpolation lower limit for κ_λ.
κ_λ_ub_default = 2.5 # interpolation upper limit for κ_λ.

# TODO: functions for creating these config files, or at least documentation about it.
#surrogate_config_path = "/home/roy/Documents/repo/NMRData/input/select_molecules_surrogate_configs.json"

###
#type_SSParams = NMRSignalSimulator.getSpinSysParamsdatatype(NMRSignalSimulator.SharedShift{Float64})
type_SSParams = NMRSignalSimulator.getSpinSysParamsdatatype(NMRSignalSimulator.CoherenceShift{Float64})

# u_min = ppm2hzfunc(-0.5)
# u_max = ppm2hzfunc(4.0)

Bs, MSS, itp_samps = NMRSignalSimulator.fitclproxies(type_SSParams, As, λ0;
    names = molecule_entries,
    #config_path = surrogate_config_path,
    Δcs_max_scalar_default = Δcs_max_scalar_default,
    κ_λ_lb_default = κ_λ_lb_default,
    κ_λ_ub_default = κ_λ_ub_default,
    # u_min = u_min,
    # u_max = u_max,
    Δr_default = Δr_default,
    Δκ_λ_default = Δκ_λ_default)

# 
### plot.

## suspect there is a bug in the direct manipulation code below. Use importmodel!() instead.
#= # purposely distort the spectra by assigning random values to model parameters.
if type_SSParams == NMRSignalSimulator.SpinSysParams{NMRSignalSimulator.CoherenceShift{Float64}, NMRSignalSimulator.CoherencePhase{Float64}, NMRSignalSimulator.SharedT2{Float64}}

    B = Bs[1]
    A = As[1]
    if type_SSParams <: NMRSignalSimulator.SharedShift
        B.ss_params.shift.var[:] = rand(length(B.ss_params.shift.var))
    elseif type_SSParams <: NMRSignalSimulator.CoherenceShift
        B.ss_params.shift.var[:] = collect( rand(length(B.ss_params.shift.var[i])) .* (2*π) for i in eachindex(B.ss_params.shift.var) )
    end
    B.ss_params.T2.var[:] = rand(length(B.ss_params.T2.var)) .+ 1
    B.ss_params.phase.var[:] = collect( rand(length(B.ss_params.phase.var[i])) .* (2*π) for i in eachindex(B.ss_params.phase.var) )

    NMRSignalSimulator.resolveparameters!(B.ss_params.phase, A.Δc_bar)
    NMRSignalSimulator.resolveparameters!(B.ss_params.shift, A.Δc_bar)
end =#

## modify such that the phase of a spin system is non-zero.

# debug.
shifts, phases, T2s = MSS.shifts, MSS.phases, MSS.T2s
mapping = NMRSignalSimulator.getParamsMapping(shifts, phases, T2s)
# end debug.

model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = copy(w_oracle))
x_oracle = copy(model_params.var_flat)

n_select = 1
sys_select = 1
st = model_params.systems_mapping.phase.st[n_select][sys_select]
fin = model_params.systems_mapping.phase.fin[n_select][sys_select]
x_oracle[st:fin] = randn(fin-st+1)
model_params.var_flat[:] = x_oracle
NMRSignalSimulator.importmodel!(model_params)
###  end modification of phase.

f = uu->NMRSignalSimulator.evalclmixture(uu, As, Bs; w = w_oracle)

hz2ppmfunc = uu->(uu - ν_0ppm)*SW/fs
ppm2hzfunc = pp->(ν_0ppm + pp*fs/SW)

# test params.
ΩS_ppm = collect( hz2ppmfunc.( NMRSignalSimulator.combinevectors(A.Ωs) ./ (2*π) ) for A in As )
ΩS_ppm_flat = NMRSignalSimulator.combinevectors(ΩS_ppm)
P_max = maximum(ΩS_ppm_flat) + 0.5
P_min = minimum(ΩS_ppm_flat) - 0.5

P = LinRange(P_min, P_max, 80000)
#P = LinRange(-0.2, 5.5, 80000)
U = ppm2hzfunc.(P)
U_rad = U .* (2*π)

## parameters that affect qs.
# A.ζ, A.κs_λ, A.κs_β
# A.ζ_singlets, A.αs_singlets, A.Ωs_singlets, A.β_singlets, A.λ0, A.κs_λ_singlets
q = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Bs; w = w_oracle)

#q = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Es)

f_U = f.(U_rad)
q_U = q.(U_rad)

# for downstream examples.
S_U = copy(q_U)

discrepancy = abs.(f_U-q_U)
max_val, ind = findmax(discrepancy)
println("relative discrepancy = ", norm(discrepancy)/norm(f_U))
println("max discrepancy: ", max_val)
println()

# ## remove areas with low signal from plotting to reduce plot size.
# reduction_factor = 100
# threshold_factor =  α_relative_lower_threshold/10
# inds, keep_inds, inds_for_reducing = NMRSignalSimulator.prunelowsignalentries(q_U, threshold_factor, reduction_factor)
#
# q_U_display = q_U[inds]
# f_U_display = f_U[inds]
# P_display = P[inds]

q_U_display = q_U
f_U_display = f_U
P_display = P

## visualize.
PyPlot.figure(fig_num)
fig_num += 1

# PyPlot.plot(P, real.(f_U), label = "f")
# PyPlot.plot(P, real.(q_U), label = "q")
PyPlot.plot(P_display, real.(f_U_display), label = "f")
PyPlot.plot(P_display, real.(q_U_display), label = "q")
PyPlot.plot(P_display, real.(q_U_display), "x")

PyPlot.legend()
PyPlot.xlabel("ppm")
PyPlot.ylabel("real")
PyPlot.title("f vs q")


## visualize. zoom in.

inds = findall(xx->(2.5<xx<3.9), P_display)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P_display[inds], real.(f_U_display[inds]), label = "f")
PyPlot.plot(P_display[inds], real.(q_U_display[inds]), label = "q")

PyPlot.legend()
PyPlot.xlabel("ppm")
PyPlot.ylabel("real")
PyPlot.title("f vs q")


# using BenchmarkTools
#
# m = 1
# A = As[1];
#
# println("qs[i][k], gs eval:")
# r0 = 2*π*U[m] - A.ss_params.shift.var[1]
# @btime A.qs[1][1](r0, 1.0)
# @btime A.gs[1][1](r0, 1.0)
#
# println("q eval.")
# @btime q.(U_rad[m]);
#
# println("q_U eval")
# @btime q.(U_rad);
# println()

## next, derivatives of q over the bounds of d and β, not κ_d and κ_β.


#q_U = q.(U_rad)

model_params = NMRSignalSimulator.MixtureModelParameters(MSS; w = copy(w_oracle))

# back up the current model parameters.
NMRSignalSimulator.exportmodel!(model_params)
x0 = copy(model_params.var_flat)
@assert norm(x0-x_oracle) < 1e-12

println()



lbs, ubs = NMRSignalSimulator.fetchbounds(model_params, Bs; shift_proportion = 0.9)

if typeof(first(shifts)) <: NMRSignalSimulator.CoherenceShift
    ζ_max = Bs[3].op_range.d_max .* (2*π)
    Δcs_max = NMRSignalSimulator.ζ2Δcs(ζ_max[1], ν_0ppm, hz2ppmfunc)

    # move DSS to see effect of ζ.
    Bs[3].ss_params.shift.ζ[1][1] # 0.65 ppm group.
    Bs[3].ss_params.shift.ζ[2][1] # 0 ppm singlet.

    #Bs[3].ss_params.shift.ζ[1][1] = -0.1
    Bs[3].ss_params.shift.ζ[1][1] = NMRSignalSimulator.Δcs2ζ(-0.1, ppm2hzfunc)
    inds = findall(xx->(0.4<xx<0.8), P_display)

    #Bs[3].ss_params.shift.ζ[2][1] = NMRSignalSimulator.Δcs2ζ(-0.123, ppm2hzfunc)
    #inds = findall(xx->(-0.2<xx<0.2), P_display)

    q_U = q.(U_rad)

    PyPlot.figure(fig_num)
    fig_num += 1

    PyPlot.plot(P_display[inds], real.(f_U[inds]), label = "f")
    PyPlot.plot(P_display[inds], real.(q_U[inds]), label = "q")

    PyPlot.legend()
    PyPlot.xlabel("ppm")
    PyPlot.ylabel("real")
    PyPlot.title("zoomed in")

    # revert back to oracle values.

    model_params.var_flat[:] = x_oracle
    NMRSignalSimulator.importmodel!(model_params)

    f_U = f.(U_rad)
    q_U = q.(U_rad)
    discrepancy = abs.(f_U-q_U)
    max_val, ind = findmax(discrepancy)
    println("relative discrepancy = ", norm(discrepancy)/norm(f_U))
    println("max discrepancy: ", max_val)
    println()
end