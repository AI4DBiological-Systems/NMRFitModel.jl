
Random.seed!(25)

include("./helpers/utils.jl")

# cs_sys_mixture, cs_singlets_mixture = NMRHamiltonian.extractcs(Phys)
# As

#extractphysicalparameters = NMRSignalSimulator.extractphysicalparameters
#updateshiftphysicalparameters! = NMRSignalSimulator.updateshiftphysicalparameters!
updatephysicalparameters! = NMRSignalSimulator.updatephysicalparameters!
ζ2Δcs = NMRSignalSimulator.ζ2Δcs
createtablecolumns = NMRSignalSimulator.createtablecolumns
#extractparameters = NMRSignalSimulator.extractparameters
addchemicalshifts = NMRSignalSimulator.addchemicalshifts

serializclproxies = NMRSignalSimulator.serializclproxies
deserializclproxies = NMRSignalSimulator.deserializclproxies

shifts, phases, T2s, Δc_bars = MSS.shifts, MSS.phases, MSS.T2s, MSS.Δc_bars


n_select = 1
A = As[n_select]
phy = Phys[n_select]
@show propertynames(phy)

i_select = 1
ηs = A.Δc_bar[i_select]

## set parameters to a random vector.
#x_star = generateparameters(lbs, ubs)

NMRSignalSimulator.importmodel!(model_params, x_star)

## set all parameters to zeros.
# p_zero = zeros(length(x_star))
# NMRSignalSimulator.importmodel!(model_params, p_zero)

#Phys2 = updatechemshifts(Phys, shifts, ν_0ppm, hz2ppmfunc)

Phys_shift = deepcopy(Phys)
#updateshiftphysicalparameters!(Phys_shift, shifts, fs, SW, ν_0ppm)
updatephysicalparameters!(
    Phys_shift,
    shifts,
    vv->ζ2Δcs(vv, ν_0ppm, hz2ppmfunc),
    Δc_bars;
    var_tag = "shift")

Phys_phase = deepcopy(Phys)
updatephysicalparameters!(
    Phys_phase, phases, identity, Δc_bars; var_tag = "phase")

Phys_T2 = deepcopy(Phys)
updatephysicalparameters!(Phys_T2, T2s, identity, Δc_bars;
    var_tag = "T2")

nucleus_ID_set, location_set, cs_set = createtablecolumns(Phys)
nucleus_ID_set, location_set, shift_set = createtablecolumns(Phys_shift)
nucleus_ID_set2, location_set2, phase_set = createtablecolumns(Phys_phase)
nucleus_ID_set3, location_set3, T2_set = createtablecolumns(Phys_T2)

# new_Phys = addchemicalshifts(Phys, Phys_shift, 0.1)
# nucleus_ID_set, location_set, new_cs__set = createtablecolumns(new_Phys)

entries_set = collect( molecule_entries[ location_set[n][begin] ] for n in eachindex(location_set) )
spin_sys_set = collect( location_set[n][end] for n in eachindex(location_set) )

w_set = collect( w[findfirst(xx->xx==entries_set[i], molecule_entries)] for i in eachindex(entries_set) )

@assert nucleus_ID_set == nucleus_ID_set2
@assert location_set == location_set2

using TypedTables

tab = Table(
    entry = entries_set,
    spin_sys = spin_sys_set,
    nucleus = nucleus_ID_set,
    N_nuclei = length.(nucleus_ID_set),
    cs = cs_set, # ppm
    #new_cs = new_cs__set, # ppm
    shift_ζ = shift_set, # ppm
    phase_κ = phase_set, # radians
    T2_ξ = T2_set, # dimensionless multiplier.
    weight = w_set
)
@show λ0





#### From a SharedShift model to a coherence shift model.
# Method 1: epoch. save SharedShift model as Phy, and resimulate As, Bs.
# Method 2: average. Without converting to As. Carry the Phase and T2 models and current parameters over to the converted model.

# for method 1, use the approach earlier in this script to save new Phys variables, then resimulate As, Bs.

if typeof(first(shifts)) <: NMRSignalSimulator.SharedParams
    # the following is for method 2.

    # typical code for serialize and deserialize. Verify.
    S_Bs = serializclproxies(Bs)
    ss_params_set2, op_range_set2, λ02 = deserializclproxies(S_Bs)

    ss_params3 = NMRSignalSimulator.convertmodel(ss_params_set2, As)

    Bs2, MSS2 = NMRSignalSimulator.recoverclproxies(
        itp_samps,
        #ss_params_set2, # sharedshift. produces same Bs2 as Bs.
        ss_params3, # converted coherenceshift. produces different Bs2.
        op_range_set2,
        As,
        λ02,
    )

    # model_params2.var_flat is a flat vector of parameters.
    model_params2 = NMRSignalSimulator.MixtureModelParameters(MSS2; w = copy(w_oracle))

    #@assert 1==2
    w = w_oracle
    q = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Bs; w = w)
    q2 = uu->NMRSignalSimulator.evalclproxymixture(uu, As, Bs2; w = w)

    q_U = q.(U_rad)
    q2_U = q2.(U_rad)
    #@test norm(q_U - q2_U) < zero_tol
    @show norm(q_U - q2_U)

end
    # TODO: package this script to a test, and use molecules with ME in multiple spin systems.