"""
    convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T

converts compact domain x ∈ [a,b] to compact domain out ∈ [c,d].
"""
function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function generateparameters(lbs::Vector{T}, ubs::Vector{T})::Vector{T} where T
    
    @assert length(lbs) == length(ubs)

    return collect( convertcompactdomain(rand(), zero(T), one(T), lbs[i], ubs[i]) for i in eachindex(lbs) )
end

function combinevectors(x::Vector{Vector{T}})::Vector{T} where T

    if isempty(x)
        return Vector{T}(undef, 0)
    end

    N = sum(length(x[i]) for i in eachindex(x))

    y = Vector{T}(undef,N)

    st_ind = 0
    fin_ind = 0
    for i in eachindex(x)
        st_ind = fin_ind + 1
        fin_ind = st_ind + length(x[i]) - 1

        y[st_ind:fin_ind] = x[i]
    end

    return y
end

###### specify region.
function getΩS(As::Vector{NMRHamiltonian.SHType{T}}) where T

    ΩS = Vector{Vector{Vector{T}}}(undef, length(As))

    for n in eachindex(As)

        ΩS[n] = Vector{Vector{T}}(undef, length(As[n].Ωs))
        for i in eachindex(As[n].Ωs)

            ΩS[n][i] = copy(As[n].Ωs[i])

        end
    end

    return ΩS
end

function getPs( ΩS::Vector{Vector{Vector{T}}}, hz2ppmfunc) where T <: Real

    N_compounds = length(ΩS)

    Ps = Vector{Vector{Vector{T}}}(undef, N_compounds)
    for n = 1:N_compounds

        Ps[n] = Vector{Vector{T}}(undef, length(ΩS[n]))
        for i in eachindex(ΩS[n])

            Ps[n][i] = Vector{T}(undef, length(ΩS[n][i]))
            for l in eachindex(ΩS[n][i])

                Ps[n][i][l] = hz2ppmfunc( ΩS[n][i][l]/(2*π) )
            end
        end
    end

    return Ps
end

function getPsnospininfo(As::Vector{NMRSignalSimulator.SHType{T}}, hz2ppmfunc) where T

    ΩS_ppm = Vector{Vector{T}}(undef, length(As))

    for (n,A) in enumerate(As)

        ΩS_ppm[n] = hz2ppmfunc.( combinevectors(A.Ωs) ./ (2*π) )

    end

    return ΩS_ppm
end

function initializeΔsyscs(As, x::T) where T

    N_molecules =
    Δsys_cs = Vector{Vector{T}}(undef,  length(As))

    for n in eachindex(As)

        N_sys = length(As[n].N_spins_sys)
        Δsys_cs[n] = Vector{T}(undef, N_sys)

        for i in eachindex(Δsys_cs[n])
            Δsys_cs[n][i] = x
        end

        # for _ in eachindex(As[n].αs_singlets)
        #     push!(Δsys_cs[n], x)
        # end
    end

    return Δsys_cs
end

###### load experiments.

# for use with NMRglue. move out of this library eventually, into a Python script.
function extractexperimentparams(dic)

    #DE = dic["acqus"]["DE"] # in microseconds. dead time.
    #BF1 = dic["acqus"]["BF1"]

    ## see Bruker TopSpin acquistion commands and parameters v 003.
    # TD - Time Domain; Number Of Raw Data Points. i.e., length(data)*2 - TD = 0.
    # total samples from both time-series.
    TD = dic["acqus"]["TD"]

    # SW - Spectral Width in ppm
    SW = dic["acqus"]["SW"]

    # SFO1 - SFO8 - Irradiation (carrier) Frequencies For Channels f1 to f8
    SFO1 = dic["acqus"]["SFO1"]

    O1 = dic["acqus"]["O1"]

    fs_dic = dic["acqus"]["SW_h"]

    return TD, SW, SFO1, O1, fs_dic
end

# function saveloadBSON(file_path, S)

#     BSON.bson(file_path, S)
#     return BSON.load(file_path)
# end

# function saveloadJSON(file_path, S)

#     NMRHamiltonian.saveasJSON(file_path, S)
#     return JSON3.read(read(file_path))
# end




########## debug.

# BatchEvalBuffer = NMRSignalSimulator.BatchEvalBuffer
# MixtureSpinSys = NMRSignalSimulator.MixtureSpinSys

# function constructdesignmatrix!(
#     X::BatchEvalBuffer{T},
#     MSS::MixtureSpinSys,
#     )::Matrix{T} where T <: AbstractFloat

#     phases = MSS.phases
#     A, re_evals, im_evals = X.A, X.re_evals, X.im_evals

#     N = getNentries(MSS)
#     @assert size(A) == (length(X.U_rad), N)
#     @assert length(re_evals) == length(im_evals) == N

#     fill!(A, zero(Complex{T}))

#     for n in axes(A,2)

#         for i in eachindex(X.re_evals[n])
#             for k in eachindex(X.re_evals[n][i])
                
#                 for m in axes(A,1)

#                     A[m,n] += Complex(
#                         re_evals[n][i][k][m],
#                         im_evals[n][i][k][m],
#                     )*cis(phases[n].β[i][k])
#                 end
#             end
#         end
#     end
#     @show A[1]
    
#     B = reinterpret(T, A)
#     return B
# end

# function getNentries(MSS::MixtureSpinSys)::Int

#     return length(MSS.Δc_bars)
# end

################# multi-start.

function getinitialiterates(lbs::Vector{T}, ubs::Vector{T}, N::Int) where T

    s = Sobol.SobolSeq(lbs, ubs)

    X = Vector{Vector{T}}(undef, N)
    for n in eachindex(X)
        X[n] = Sobol.next!(s)
    end
    
    return X
end


