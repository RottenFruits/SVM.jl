module SVM

using StatsBase, LinearAlgebra, Printf
import StatsBase: predict

export svm, cddual, pegasos, predict

mutable struct SVMFit
    w::Vector{Float64}
    pass::Int
    converged::Bool
end

function Base.show(io::IO, fit::SVMFit)
    @printf(io, "Fitted linear SVM\n")
    @printf(io, " * Non-zero weights: %d\n", sum(model.w .!= 0))
    @printf(io, " * Iterations: %d\n", fit.pass)
    @printf(io, " * Converged: %s\n", string(fit.converged))
end

function predict(fit::SVMFit, X::AbstractMatrix)
    n, l = size(X)
    preds = Array{Float64}(undef, l)
    for i in 1:l
        tmp = 0.0
        for j in 1:n
            tmp += fit.w[j] * X[j, i]
        end
        preds[i] = sign(tmp)
    end
    return preds
end

include("pegasos.jl")
include("cddual.jl")

function svm(X::AbstractMatrix,
                      Y::AbstractVector;
                      k::Integer = 5,
                      lambda::Real = 0.1,
                      maxpasses::Integer = 100)
    pegasos(X, Y, k = k, lambda = lambda, maxpasses = maxpasses)
end

end # module
