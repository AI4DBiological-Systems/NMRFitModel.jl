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