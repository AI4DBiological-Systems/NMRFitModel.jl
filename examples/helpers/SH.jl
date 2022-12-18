function runSH(
    Phys,
    fs::T,
    SW::T,
    ν_0ppm::T,
    molecule_entries::Vector{String},
    max_partition_size_offset::Int;
    search_θ = true,
    θ_default = 0.0,
    γ_base = 0.1,
    γ_rate = 1.05,
    max_iters_γ = 100,
    fully_connected_convex_clustering = false, #  overides all knn-related optional inputs
    max_connected_components_offset = -1,
    starting_manual_knn = 60,
    unique_cs_atol = 1e-6,
    length_scale_base = 10.0,
    length_scale_rate = 0.7,
    min_dynamic_range = 0.95,
    cc_gap_tol = 1e-8,
    cc_max_iters = 300,
    assignment_zero_tol = 1e-3,
    ) where T <: AbstractFloat


    # extract chemical shifts for spin systems and singlets.
    cs_sys_mixture, cs_singlets_mixture = NMRHamiltonian.extractcs(Phys)

    # if using default tolerances for coherence and intensity thresholding.
    mixture_sh_config = NMRHamiltonian.defaultmixtureshsconfig(cs_sys_mixture)
    mixture_parts_params = NMRHamiltonian.defaultmixturepartitionsparameters(
        cs_sys_mixture,
        θ_default
    )
    # # if loading from file.
    # file_filder = "./configs"
    # mixture_sh_config = NMRHamiltonian.loadmixtureshsconfig(
    #     cs_sys_mixture,
    #     joinpath(file_folder, "mixture_SH.json"),
    #     molecule_entries,
    #     Float64,
    # )
    # mixture_parts_params = NMRHamiltonian.loadmixturepartitionsparameters(
    #     cs_sys_mixture,
    #     joinpath(file_folder, "mixture_partition.json"),
    #     molecule_entries,
    #     2.0
    # )

    constantknnfunc, constantradiusfunc, θs,
        γs = NMRHamiltonian.setupconstantparameteroptions(molecule_entries, mixture_parts_params)

    # getgraphconfigfunc can be one of the following.:
    #   - defaultknnsearchconfig
    #   - defaultknnconfig
    #   - defaultradiusconfig
    #   - defaultradiussearchconfig
    #   - constantknnfunc
    #   - constantradiusfunc



    searchknnconfigfunc = (nn, ii, cc, aa)->NMRHamiltonian.defaultknnsearchconfig(
        nn, ii, cc, aa;
        verbose = true,
        start_knn = max(starting_manual_knn, round(Int, length(cc)*0.05)),
        max_knn = max(starting_manual_knn, round(Int, length(cc)*0.2)),
        max_connected_components_offset  = max_connected_components_offset,
    )
    if fully_connected_convex_clustering
        searchknnconfigfunc = (nn, ii, cc, aa)->NMRHamiltonian.defaultknnsearchconfig(
            nn, ii, cc, aa;
            verbose = true,
            start_knn = length(cc),
        )
    end

    getsearchθconfigfunc = NMRHamiltonian.disablesearch
    if search_θ
        getsearchθconfigfunc = (nn, ii, cc, aa)->NMRHamiltonian.createsearchθconfigs(
            nn, ii, cc, aa;
            length_scale_base = length_scale_base,
            length_scale_rate = length_scale_rate,
            min_dynamic_range = min_dynamic_range,
        )
    end
    searchγconfigfunc = (nn, ii, cc, aa)->NMRHamiltonian.createsearchγconfigs(
        nn, ii, cc, aa;
        max_partition_size_offset = max_partition_size_offset,
        γ_base = γ_base,
        γ_rate = γ_rate,
        max_iters_γ = max_iters_γ,
    )

    getassignmentfunc = (nn, ii, cc, aa)->NMRHamiltonian.defeaultassignmentconfig(
        nn, ii, cc, aa;
        assignment_zero_tol = assignment_zero_tol,
    )

    part_algs = NMRHamiltonian.generatemixturepartitionalgorithm(
        molecule_entries,
        θs,
        γs,
        Phys;
        #getgraphconfigfunc = constantradiusfunc,
        getgraphconfigfunc = NMRHamiltonian.defaultknnsearchconfig,
        getsearchθconfigfunc = getsearchθconfigfunc,
        getsearchγconfigfunc = searchγconfigfunc,
        getassignmentfunc = getassignmentfunc,
        report_cost = true,
        verbose_kernel = true,
        gap_tol = cc_gap_tol,
        max_iters = cc_max_iters,
    )

    As, Rs = NMRHamiltonian.setupmixtureSH(
        part_algs,
        molecule_entries,
        fs, SW, ν_0ppm,
        Phys,
        mixture_sh_config,
    )

    return As, Rs, Phys
end

function savePhys(Phys::Vector{PPT}) where PPT
    names = collect(fieldnames(PPT))

    for name in names
        if typeof(getfield(p, name)) <: Vector
            resize!(getfield(p, name), 0)
        end
    end
    
end