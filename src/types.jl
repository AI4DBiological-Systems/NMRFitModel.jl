
struct OSQPConfig{T}
    eps_abs::T
    eps_rel::T
    max_iter::Int
    verbose::Bool
    alpha::T
    adaptive_rho::Bool
end

function generateOSQPConfig(
    ::Type{T};
    eps_abs = 1e-12,
    eps_rel = 1e-8,
    max_iter::Int = 4000,
    verbose::Bool = false,
    alpha = 1.0,
    adaptive_rho = true,
    ) where T <: AbstractFloat
    #
    return OSQPConfig(
        convert(T, eps_abs),
        convert(T, eps_rel),
        max_iter,
        verbose,
        convert(T, alpha),
        adaptive_rho,
    )
end

struct BLSParams{T <: AbstractFloat}
    P_vec::Vector{T}
    A_buf::Matrix{Complex{T}}
    optim_problem
end


##### model fitting configs.

struct BLSCLConfig
    U_rad
    λ0
    αs
    Ωs
    τs
    βs
    ξs
end