using ParallelKMeans
using Random

include("common.jl")
include("simple_iforest.jl")

const KMEANS_CONFIGS = [
    (name = "kmeans_practical", k = 256, threshold_percentile = 91.0),
    (name = "kmeans_mid_fpr", k = 1024, threshold_percentile = 90.0),
    (name = "kmeans_max_f1", k = 1024, threshold_percentile = 84.0),
]

const IFOREST_CONFIGS = [
    (name = "iforest_practical", trees = 100, sample_size = 256, threshold_percentile = 95.0),
    (name = "iforest_balanced", trees = 200, sample_size = 512, threshold_percentile = 92.0),
    (name = "iforest_higher_recall", trees = 300, sample_size = 512, threshold_percentile = 88.0),
]

function min_distances(centers::Matrix{Float64}, points::Matrix{Float64})
    center_sq = reshape(vec(sum(abs2, centers; dims = 1)), :, 1)
    point_sq = reshape(vec(sum(abs2, points; dims = 1)), 1, :)
    cross = transpose(centers) * points
    distances_sq = center_sq .+ point_sq .- 2.0 .* cross
    distances_sq .= max.(distances_sq, 0.0)
    return sqrt.(vec(minimum(distances_sq; dims = 1)))
end

function fit_kmeans(x_train::Matrix{Float64}, k::Int; random_seed::Int = MODEL_RANDOM_SEED)
    points = permutedims(x_train)
    return ParallelKMeans.kmeans(
        ParallelKMeans.Yinyang(),
        points,
        k;
        n_threads = Threads.nthreads(),
        max_iters = 300,
        verbose = false,
        rng = MersenneTwister(random_seed),
    )
end

function run_kmeans(
    x_train::Matrix{Float64},
    x_eval::Matrix{Float64},
    eval_frame::DataFrame;
    random_seed::Int = MODEL_RANDOM_SEED,
)
    rows = NamedTuple[]
    cache = Dict{Tuple{Int, Int}, Tuple{Any, Vector{Float64}, Float64}}()
    points_train = permutedims(x_train)
    points_eval = permutedims(x_eval)

    for config in KMEANS_CONFIGS
        cache_key = (config.k, random_seed)
        if !haskey(cache, cache_key)
            started = time()
            model = fit_kmeans(x_train, config.k; random_seed = random_seed)
            train_distances = min_distances(model.centers, points_train)
            cache[cache_key] = (model, train_distances, time() - started)
        end

        model, train_distances, fit_seconds = cache[cache_key]
        threshold = quantile(train_distances, config.threshold_percentile / 100.0)
        eval_distances = min_distances(model.centers, points_eval)
        predictions = Int.(eval_distances .> threshold)
        metrics = metrics_table(eval_frame, predictions)

        push!(
            rows,
            (
                algorithm = "kmeans",
                config_name = config.name,
                parameters = "k=$(config.k), threshold_percentile=$(config.threshold_percentile)",
                precision = metrics.precision,
                recall = metrics.recall,
                f1 = metrics.f1,
                fpr = metrics.fpr,
                tp = metrics.tp,
                fp = metrics.fp,
                tn = metrics.tn,
                fn = metrics.fn,
                runtime_seconds = fit_seconds,
            ),
        )
    end

    return rows
end

function run_iforest(
    x_train::Matrix{Float64},
    x_eval::Matrix{Float64},
    eval_frame::DataFrame;
    random_seed::Int = MODEL_RANDOM_SEED,
)
    rows = NamedTuple[]

    for config in IFOREST_CONFIGS
        started = time()
        forest = SimpleIsolationForest(
            n_trees = config.trees,
            sample_size = config.sample_size,
            random_state = random_seed,
        )
        fit!(forest, x_train)
        train_scores = score_samples(forest, x_train)
        threshold = quantile(train_scores, config.threshold_percentile / 100.0)
        predictions = Int.(score_samples(forest, x_eval) .> threshold)
        metrics = metrics_table(eval_frame, predictions)

        push!(
            rows,
            (
                algorithm = "isolation_forest",
                config_name = config.name,
                parameters = "trees=$(config.trees), sample_size=$(config.sample_size), threshold_percentile=$(config.threshold_percentile)",
                precision = metrics.precision,
                recall = metrics.recall,
                f1 = metrics.f1,
                fpr = metrics.fpr,
                tp = metrics.tp,
                fp = metrics.fp,
                tn = metrics.tn,
                fn = metrics.fn,
                runtime_seconds = time() - started,
            ),
        )
    end

    return rows
end

function main()
    run_only = "all"
    output_name = "julia_experiments.csv"

    for (index, argument) in pairs(ARGS)
        if argument == "--only" && index < length(ARGS)
            run_only = ARGS[index + 1]
        elseif argument == "--output" && index < length(ARGS)
            output_name = ARGS[index + 1]
        end
    end

    train, eval_frame = load_samples()
    x_train, x_eval, used_columns = prepare_features(train, eval_frame)

    rows = NamedTuple[]
    if run_only in ("all", "kmeans")
        append!(rows, run_kmeans(x_train, x_eval, eval_frame))
    end
    if run_only in ("all", "iforest")
        append!(rows, run_iforest(x_train, x_eval, eval_frame))
    end

    result = DataFrame(rows)
    insertcols!(result, 1, :feature_count => fill(length(used_columns), nrow(result)))
    mkpath(RESULTS_DIR)
    out_path = joinpath(RESULTS_DIR, output_name)
    CSV.write(out_path, result)
    println(result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
