####### verify derivatives.

# verify ∇f! with numerical differentiation using f.
function verifygradient!(
    df_AN_eval::Vector{T},
    f::Function,
    ∇f!::Function,
    p_test::Vector{T};
    ND_accuracy_order = 8,
    verbose = false,
    ) where T

    fill!(df_AN_eval, NaN)
    ∇f!(p_test)

    df_ND = pp->FiniteDifferences.grad(FiniteDifferences.central_fdm(ND_accuracy_order, 1), f, pp)[1]
    df_ND_eval = df_ND(p_test)

    if verbose
        @show df_AN_eval, df_ND_eval
        println()
    end

    return norm(df_ND_eval - df_AN_eval)
end

function verifysystemsgradient(
    Bs,
    MSS,
    p_test::Vector{T};
    atol = 1e-3,
    ND_accuracy_order = 8,
    verbose = false,
    ) where T

    df_AN_eval = ones(T, 2) .* NaN

    discrepancy = zero(T)
    pass_flag = true

    # test each resonance group.
    for n in eachindex(Bs)
        for i in eachindex(Bs[n].qs)
            for k in eachindex(Bs[n].qs[i])

                if verbose
                    @show n,i,k
                end

                # real part of model.
                discrepancy += verifygradient!(
                    df_AN_eval,
                    pp->real(Bs[n].qs[i][k](pp...)),
                    pp->MSS.∇srs![n][i][k](df_AN_eval, pp...),
                    p_test;
                    ND_accuracy_order = ND_accuracy_order,
                    verbose = verbose,
                )

                # imaginary part of model.
                discrepancy += verifygradient!(
                    df_AN_eval,
                    pp->imag(Bs[n].qs[i][k](pp...)),
                    pp->MSS.∇sis![n][i][k](df_AN_eval, pp...),
                    p_test;
                    ND_accuracy_order = ND_accuracy_order,
                    verbose = verbose,
                )

                pass_flag = pass_flag & (discrepancy < atol)
            end
        end
    end

    return discrepancy, pass_flag
end

# returned discrepancies are norm-ed discrepancies.
function verifysimulatorgradient!(
    model_params,
    U_rad,
    x_test::Vector{T};
    atol = 1e-3,
    ND_accuracy_order = 8,
    verbose = false,
    ) where T

    # setup.
    grad = NMRSignalSimulator.MixtureModelGradient(model_params)
    
    discrepancies_re = ones(T, length(U_rad)) .* NaN
    discrepancies_im = ones(T, length(U_rad)) .* NaN
    #pass_flag = true

    # test each frequency location.
    for m in eachindex(U_rad)
        ω_test = U_rad[m]

        # analytical.
        dg = xx->NMRSignalSimulator.evalmodelgradient!(
            grad,
            model_params,
            ω_test,
            xx,
        )

        # numerical.
        g = xx->NMRSignalSimulator.evalmodel!(model_params, ω_test, xx)
        gr = xx->real(g(xx))
        gi = xx->imag(g(xx))

        if verbose
            @show m
        end

        # real part of model.
        discrepancies_re[m] = verifygradient!(
            grad.grad_re_flat,
            gr,
            dg,
            x_test;
            ND_accuracy_order = ND_accuracy_order,
            verbose = verbose,
        )

        # imaginary part of model.
        discrepancies_im[m] = verifygradient!(
            grad.grad_im_flat,
            gi,
            dg,
            x_test;
            ND_accuracy_order = ND_accuracy_order,
            verbose = verbose,
        )

        #pass_flag = pass_flag & (discrepancies[m] < atol)
    end

    return discrepancies_re, discrepancies_im
end


function verifybatchsimulatorgradient!(
    model_params,
    U_rad,
    x_test::Vector{T},
    ) where T

    mapping = model_params.systems_mapping
    w = model_params.w
    MSS = model_params.MSS

    ### compute gradient using the method under test (batch method).

    NMRSignalSimulator.importmodel!(model_params, x_test)

    gs_re, gs_im = NMRSignalSimulator.packagedevalgradient!(model_params, U_rad, x_test)

    # evaluate reference.
    refs_re = NMRSignalSimulator.createbatchgradbuffer(MSS, length(U_rad))
    refs_im = NMRSignalSimulator.createbatchgradbuffer(MSS, length(U_rad))
    for i in eachindex(refs_re)
        fill!(refs_re[i], zero(T))
        fill!(refs_im[i], zero(T))
    end


    ### compute gradient using the reference method at each frequency position in U_rad.
    
    NMRSignalSimulator.importmodel!(model_params, x_test)

    for m in eachindex(U_rad)

        grad = NMRSignalSimulator.MixtureModelGradient(model_params)

        NMRSignalSimulator.evalmodelgradient!(
            grad,
            model_params,
            U_rad[m],
            x_test,
        )

        # update storage.
        for l in eachindex(grad.grad_re_flat)
            refs_re[l][m] = grad.grad_re_flat[l]
        end

        for l in eachindex(grad.grad_im_flat)
            refs_im[l][m] = grad.grad_im_flat[l]
        end

    end

    # compute discrepancy.
    discrepancies_re = ones(T, length(U_rad), length(refs_re)) .* NaN
    discrepancies_im = ones(T, length(U_rad), length(refs_im)) .* NaN

    for m in axes(discrepancies_re,1)
        for l in axes(discrepancies_re,2)
            discrepancies_re[m,l] = abs(refs_re[l][m] - gs_re[l][m])
        end
    end

    for m in axes(discrepancies_im,1)
        for l in axes(discrepancies_im,2)
            discrepancies_im[m,l] = abs(refs_im[l][m] - gs_im[l][m])
        end
    end
 
    return discrepancies_re, discrepancies_im,
        refs_re, refs_im, gs_re, gs_im
end

function verifycostfunc(
    model_params,
    lbs,
    ubs,
    y::Vector{Complex{T}},
    U_rad,
    N_p_tests::Int,
    ) where T <: AbstractFloat

    # constants.
    mapping, MSS = model_params.systems_mapping, model_params.MSS
    
    X, gs_re, gs_im, shift_multi_inds, phase_multi_inds, T2_multi_inds = NMRSignalSimulator.costfuncsetup(mapping, MSS, U_rad)
    C = NMRSignalSimulator.CostFuncBuffer(X, gs_re, gs_im, shift_multi_inds, phase_multi_inds, T2_multi_inds)

    # oracle.
    costfunc0 = pp->NMRSignalSimulator.evalcost(model_params, pp, y, U_rad)

    # method under test.
    costfunc! = pp->NMRSignalSimulator.evalcost!(
        grad_cost,
        model_params,
        pp,
        C,
        y,
    )
    
    # buffers.
    grad_cost = ones(T, NMRSignalSimulator.getNvars(model_params))

    # tests.
    discrepancies = ones(T, N_p_tests) .* NaN
    ps = Vector{Vector{T}}(undef, N_p_tests)
    for n = 1:N_p_tests
        
        p_test = generateparameters(lbs, ubs)

        c_oracle = costfunc0(p_test)

        c = costfunc!(p_test)

        discrepancies[n] = verifygradient!(grad_cost, costfunc0, costfunc!, p_test)

        #@test discrepancy < atol
        ps[n] = p_test
    end

    return discrepancies, ps
end