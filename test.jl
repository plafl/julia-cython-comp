include("model.jl")


using Serialization
using Random
using SparseArrays
using Printf
using Statistics
using .Model

println("Loading data...")
X = deserialize("lastfm.jlser")
interactions = Interactions(X)


function select_k(scores::Matrix{<:AbstractFloat}, k::Int)
    n = size(scores, 2)
    y = Array{Int}(undef, k, n)
    ix = Array{Int}(undef, size(scores, 1))
    for i=1:n
        partialsortperm!(ix, view(scores, :, i), k, rev=true)
        y[:, i] = view(ix, 1:k)
    end
    return y
end

function precision_at_k(model::Factorization,
                        interactions::SparseMatrixCSC,
                        users::Vector{Int},
                        k::Int)
    scores = model.item_bias .+ model.item_embeddings' * model.user_embeddings[:, users]
    recommendations = select_k(scores, k)
    pk = zeros(Float32, length(users))
    for i=1:size(recommendations, 2)
        for j=1:size(recommendations, 1)
            user = users[i]
            item = recommendations[j, i]
            if is_positive(interactions, user, item)
                pk[i] += 1.0f0
            end
        end
    end
    return pk / k
end


println("Creating and initializing model")
model = Factorization{Float32}(size(X, 2), size(X, 1), 32)
optimizer = FactorizationAdadelta(model, 1e-6, 0.95)

users_val = abs.(rand(Int, 1000)) .% size(interactions.sparse, 2) .+ 1
println("Training")
init!(interactions, model)
for i=1:5
    t1 = time_ns()
    epoch!(model, optimizer, interactions, 1000)
    t2 = time_ns()
    @printf("Epoch time: %.2fmin ", (t2 - t1)/1e9/60)
    @printf("precision@5: %.3f\n",
            mean(precision_at_k(model, interactions.sparse, users_val, 5)))
end
