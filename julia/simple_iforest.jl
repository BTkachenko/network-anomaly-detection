using Random
using Statistics

mutable struct IsolationNode
    size::Int
    feature::Int
    threshold::Float64
    left::Union{Nothing, IsolationNode}
    right::Union{Nothing, IsolationNode}
    is_leaf::Bool
end

function leaf_node(size::Int)
    return IsolationNode(size, 0, 0.0, nothing, nothing, true)
end

function split_node(size::Int, feature::Int, threshold::Float64, left::IsolationNode, right::IsolationNode)
    return IsolationNode(size, feature, threshold, left, right, false)
end

function average_path_length(sample_size::Int)
    if sample_size <= 1
        return 0.0
    elseif sample_size == 2
        return 1.0
    end

    harmonic = sum(1.0 / value for value in 1:(sample_size - 1))
    return 2.0 * harmonic - (2.0 * (sample_size - 1) / sample_size)
end

mutable struct SimpleIsolationForest
    n_trees::Int
    sample_size::Int
    max_depth::Int
    random_state::Int
    trees::Vector{IsolationNode}
end

function SimpleIsolationForest(; n_trees::Int = 100, sample_size::Int = 256, max_depth::Int = 0, random_state::Int = 2029)
    return SimpleIsolationForest(n_trees, sample_size, max_depth, random_state, IsolationNode[])
end

function fit!(forest::SimpleIsolationForest, x::Matrix{Float64})
    rng = MersenneTwister(forest.random_state)
    actual_sample_size = min(forest.sample_size, size(x, 1))
    depth_limit = forest.max_depth > 0 ? forest.max_depth : ceil(Int, log2(max(actual_sample_size, 2)))
    empty!(forest.trees)

    for _ in 1:forest.n_trees
        indices = randperm(rng, size(x, 1))[1:actual_sample_size]
        sample = x[indices, :]
        push!(forest.trees, build_tree(sample, 0, depth_limit, rng))
    end

    forest.sample_size = actual_sample_size
    forest.max_depth = depth_limit
    return forest
end

function build_tree(x::Matrix{Float64}, depth::Int, depth_limit::Int, rng::AbstractRNG)
    if depth >= depth_limit || size(x, 1) <= 1
        return leaf_node(size(x, 1))
    end

    mins = vec(minimum(x; dims = 1))
    maxs = vec(maximum(x; dims = 1))
    valid_features = findall(maxs .> mins)
    isempty(valid_features) && return leaf_node(size(x, 1))

    feature = rand(rng, valid_features)
    threshold = rand(rng) * (maxs[feature] - mins[feature]) + mins[feature]
    left_mask = x[:, feature] .< threshold
    right_mask = .!left_mask

    if !any(left_mask) || !any(right_mask)
        return leaf_node(size(x, 1))
    end

    left = build_tree(x[left_mask, :], depth + 1, depth_limit, rng)
    right = build_tree(x[right_mask, :], depth + 1, depth_limit, rng)
    return split_node(size(x, 1), feature, threshold, left, right)
end

function path_length(row::AbstractVector{Float64}, node::IsolationNode, depth::Int)
    if node.is_leaf
        return depth + average_path_length(node.size)
    end

    next_node = row[node.feature] < node.threshold ? node.left : node.right
    @assert next_node !== nothing
    return path_length(row, next_node, depth + 1)
end

function score_samples(forest::SimpleIsolationForest, x::Matrix{Float64})
    isempty(forest.trees) && error("Model musi zostac wytrenowany przed score_samples().")
    scores = zeros(Float64, size(x, 1))
    normalizer = average_path_length(forest.sample_size)
    normalizer == 0.0 && return scores

    for tree in forest.trees
        for row_index in 1:size(x, 1)
            scores[row_index] += path_length(view(x, row_index, :), tree, 0)
        end
    end

    average_depth = scores ./ forest.n_trees
    return 2.0 .^ (-average_depth ./ normalizer)
end
