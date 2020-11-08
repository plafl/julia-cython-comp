module Model

using Random
using SparseArrays


struct Factorization{T <: AbstractFloat}
    user_embeddings:: Matrix{T}
    item_embeddings:: Matrix{T}
    item_bias:: Vector{T}

    function Factorization{T}(n_users::Int,
                              n_items::Int,
                              n_components::Int) where {T<:AbstractFloat}
        return new{T}(
            zeros(T, n_components, n_users),
            zeros(T, n_components, n_items),
            zeros(T, n_items))
    end
end

n_users(model::Factorization) = size(model.user_embeddings, 2)
n_items(model::Factorization) = size(model.item_embeddings, 2)
n_components(model::Factorization) = size(model.user_embeddings, 1)

function(model::Factorization)(user::Int, item::Int)
    s = model.item_bias[item]
    for i=1:n_components(model)
        s += model.item_embeddings[i, item] * model.user_embeddings[i, user]
    end
    return s
end

warp_constant(model::Factorization{T}, n) where {T<:AbstractFloat} =
    convert(T, log((n_items(model) - 1) / n))

function warp_embeddings_grad(model::Factorization,
                              user::Int,
                              item_p::Int,
                              item_n::Int,
                              i::Int)
    return (model.item_embeddings[i, item_n] - model.item_embeddings[i, item_p],
            -model.user_embeddings[i, user],
            +model.user_embeddings[i, user])

end


abstract type Optimizer{T, N} end


struct SGD{T <: AbstractFloat, N} <: Optimizer{T, N}
    θ::Array{T, N}
    ϵ::T
    SGD(θ::Array{T}, ϵ::AbstractFloat) where {T <: AbstractFloat, N} =
        new{T, N}(θ, convert(T, ϵ))
end

function step!(optimizer::SGD{T, N},
               g::T,
               i::Vararg{Int, N}) where {T <: AbstractFloat, N}
    optimizer.θ[i...] -= optimizer.ϵ * g
end


struct Adadelta{T <: AbstractFloat, N} <: Optimizer{T, N}
    θ::Array{T, N}
    Eg²::Array{T, N}
    EΔ²::Array{T, N}
    ϵ::T
    η::T
    function Adadelta(θ::Array{T, N},
                      ϵ::AbstractFloat,
                      η::AbstractFloat) where {T <: AbstractFloat, N}
        return new{T, N}(θ,
                         zero(θ),
                         zero(θ),
                         convert(T, ϵ),
                         convert(T, η))
    end
end

function step!(optimizer::Adadelta{T, N},
               g::T,
               i::Vararg{Int, N}) where {T<:AbstractFloat, N}
    EΔ²i = optimizer.EΔ²[i...]
    Eg²i = optimizer.Eg²[i...]
    ϵ = optimizer.ϵ
    η = optimizer.η

    Eg²i = η*Eg²i + (1 - η)*g^2
    Δi = -sqrt((EΔ²i + ϵ)/(Eg²i + ϵ))*g
    EΔ²i = η*EΔ²i + (1 - η)*Δi^2

    optimizer.θ[i...] += Δi
    optimizer.EΔ²[i...] = EΔ²i
    optimizer.Eg²[i...] = Eg²i
end


struct Adagrad{T <: AbstractFloat, N} <: Optimizer{T, N}
    θ::Array{T, N}
    G::Array{T, N}
    ϵ::T
    η::T
    function Adagrad(θ::Array{T},
                     ϵ::AbstractFloat,
                     η::AbstractFloat) where {T <: AbstractFloat, N}
        return new{T, N}(θ,
                         zero(θ),
                         convert(T, ϵ),
                         convert(T, η))
    end
end

function step!(optimizer::Adagrad{T, N},
               g::T,
               i::Vararg{Int, N}) where {T<:AbstractFloat, N}
    Gi = optimizer.G[i...]
    ϵ = optimizer.ϵ
    η = optimizer.η

    Gi += g^2
    Δi = -η/sqrt(Gi + ϵ)*g

    optimizer.θ[i...] += Δi
    optimizer.G[i...] = Gi
end


abstract type FactorizationOptimizer{T <: AbstractFloat} end

macro optimizertype(OptimizerType, params...)
    structname = Symbol("Factorization", OptimizerType)
    return quote
        struct $structname{T} <: FactorizationOptimizer{T}
            user_embeddings::$OptimizerType{T, 2}
            item_embeddings::$OptimizerType{T, 2}
            item_bias::$OptimizerType{T, 1}

            function $structname(model::Factorization{T},
                                 $(params...)) where {T}
                return new{T}(
                    $OptimizerType(model.user_embeddings, $(params...)),
                    $OptimizerType(model.item_embeddings, $(params...)),
                    $OptimizerType(model.item_bias, $(params...)))
            end
        end
    end
end

@optimizertype(SGD, ϵ)
@optimizertype(Adadelta, ϵ, η)
@optimizertype(Adagrad, ϵ, η)

function warpstep!(model::Factorization{T},
                   optimizer::FactorizationOptimizer{T},
                   user::Int,
                   item_p::Int,
                   item_n::Int,
                   n::Int) where {T}
    C = warp_constant(model, n)
    for i=1:n_components(model)
        gu, gp, gn = warp_embeddings_grad(model, user, item_p, item_n, i)
        step!(optimizer.user_embeddings, C*gu, i, user)
        step!(optimizer.item_embeddings, C*gp, i, item_p)
        step!(optimizer.item_embeddings, C*gn, i, item_n)
    end
    step!(optimizer.item_bias, convert(T, -1), item_p)
    step!(optimizer.item_bias, convert(T, +1), item_n)
end


struct Interactions{S}
    sparse::SparseMatrixCSC{S, Int}
    users::Vector{Int}

    function Interactions(interactions::SparseMatrixCSC{S, Int}) where {S}
        n_interactions = length(interactions.nzval)
        users = Array{Int}(undef, length(interactions.nzval))
        j = 1
        for i=1:length(interactions.colptr)-1
            for j=interactions.colptr[i]:(interactions.colptr[i+1] - 1)
                users[j] = i
            end
        end
        return new{S}(interactions, users)
    end
end

function init!(interactions::Interactions, model::Factorization)
    interactions = interactions.sparse

    fill!(model.item_bias, 0)
    for i=1:length(interactions.nzval)
        model.item_bias[interactions.rowval[i]] += 1
    end
    model.item_bias ./= sum(model.item_bias)

    scale = 1.0 / n_components(model)
    rand!(model.user_embeddings)
    rand!(model.item_embeddings)
    model.user_embeddings .-= 0.5
    model.item_embeddings .-= 0.5
    model.user_embeddings .*= scale
    model.item_embeddings .*= scale
end


function is_positive(interactions::SparseMatrixCSC{S, Int},
                     user::Int,
                     item::Int) where {S}
    for i=interactions.colptr[user]:interactions.colptr[user+1] - 1
        if item == interactions.rowval[i]
            return true
        end
    end
    return false
end

function find_inversion(model::Factorization,
                        user::Int,
                        item::Int,
                        interactions::SparseMatrixCSC{S, Int},
                        max_sample::Int) where {S}
    n = n_items(model)
    s = model(user, item)
    for i=1:max_sample
        j = abs(rand(typeof(item)) % n) + 1
        t = model(user, j)
        if t > (s - 1) && !is_positive(interactions, user, j)
            return (i, j)
        end
    end
    return (max_sample, 0)
end

function epoch!(model::Factorization, optimizer::FactorizationOptimizer, interactions::Interactions, max_sample::Int)
    max_sample = min(n_items(model) - 1, max_sample)
    n_interactions = length(interactions.sparse.nzval)
    @Threads.threads for i in randperm(n_interactions)
        user = interactions.users[i]
        item_p = interactions.sparse.rowval[i]
        n, item_n = find_inversion(model, user, item_p, interactions.sparse, max_sample)
        if item_n > 0
            warpstep!(model, optimizer, user, item_p, item_n, n)
        end
    end
end


export Factorization
export n_users
export n_items
export n_components
export init!
export zeros

export FactorizationSGD
export FactorizationAdadelta
export FactorizationAdagrad

export Interactions
export is_positive
export epoch!

end
