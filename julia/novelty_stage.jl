using CSV
using DataFrames
using LIBSVM
using Statistics

include("common.jl")
include("simple_iforest.jl")

const IFOREST_NOVELTY_CONFIGS = [
    (name = "iforest_practical", trees = 100, sample_size = 256, threshold_percentile = 95.0),
    (name = "iforest_balanced", trees = 200, sample_size = 512, threshold_percentile = 92.0),
    (name = "iforest_higher_recall", trees = 300, sample_size = 512, threshold_percentile = 88.0),
]

const OCSVM_CONFIGS = [
    (name = "ocsvm_practical", nu = 0.08, gamma = 0.10),
    (name = "ocsvm_balanced", nu = 0.10, gamma = 0.10),
    (name = "ocsvm_higher_recall", nu = 0.12, gamma = 0.10),
]

const IFOREST_THRESHOLD_TREND = [
    (
        algorithm = "isolation_forest",
        setting = "trees=300,sample=512",
        trees = 300,
        sample_size = 512,
        threshold_percentile = value,
    ) for value in (95.0, 92.0, 90.0, 88.0, 86.0, 84.0)
]

const OCSVM_NU_TREND = [
    (algorithm = "one_class_svm", setting = "gamma=0.02", nu = value, gamma = 0.02)
    for value in (0.02, 0.05, 0.10, 0.15, 0.20)
]

const OCSVM_GAMMA_TREND = [
    (algorithm = "one_class_svm", setting = "nu=0.10", nu = 0.10, gamma = value)
    for value in (0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.10)
]

const OCSVM_REFINED_GRID = [
    (algorithm = "one_class_svm", setting = "local_grid", nu = nu, gamma = gamma)
    for nu in (0.08, 0.10, 0.12, 0.15) for gamma in (0.02, 0.03, 0.05, 0.07, 0.10)
]

function per_class_recall(eval_frame::DataFrame, predictions::Vector{Int}, config_name::String)
    rows = NamedTuple[]
    attack_only = copy(eval_frame[Int.(eval_frame.binary_label) .== 1, :])
    attack_only[!, :prediction] = predictions[Int.(eval_frame.binary_label) .== 1]

    for frame in groupby(attack_only, :coarse_label)
        label = String(first(frame.coarse_label))
        truth = Int.(frame.binary_label)
        pred = Int.(frame.prediction)
        tp = sum((truth .== 1) .& (pred .== 1))
        fn = sum((truth .== 1) .& (pred .== 0))
        recall = (tp + fn) == 0 ? 0.0 : tp / (tp + fn)
        push!(rows, (config_name = config_name, attack_class = label, recall = recall))
    end

    return rows
end

function evaluate_iforest(
    x_train::Matrix{Float64},
    x_eval::Matrix{Float64},
    eval_frame::DataFrame,
    config;
    random_seed::Int = MODEL_RANDOM_SEED,
)
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

    row = (
        precision = metrics.precision,
        recall = metrics.recall,
        f1 = metrics.f1,
        fpr = metrics.fpr,
        tp = metrics.tp,
        fp = metrics.fp,
        tn = metrics.tn,
        fn = metrics.fn,
        runtime_seconds = time() - started,
    )
    return row, predictions
end

function evaluate_ocsvm(
    x_train::Matrix{Float64},
    x_eval::Matrix{Float64},
    eval_frame::DataFrame,
    config;
    random_seed::Int = MODEL_RANDOM_SEED,
)
    _ = random_seed
    started = time()
    model = LIBSVM.OneClassSVM(nu = config.nu, gamma = config.gamma)
    LIBSVM.fit!(model, x_train)
    predictions = Int.(.!LIBSVM.predict(model, x_eval))
    metrics = metrics_table(eval_frame, predictions)

    row = (
        precision = metrics.precision,
        recall = metrics.recall,
        f1 = metrics.f1,
        fpr = metrics.fpr,
        tp = metrics.tp,
        fp = metrics.fp,
        tn = metrics.tn,
        fn = metrics.fn,
        runtime_seconds = time() - started,
    )
    return row, predictions
end

function run_final_configs(x_train::Matrix{Float64}, x_eval::Matrix{Float64}, eval_frame::DataFrame)
    experiment_rows = NamedTuple[]
    recall_rows = NamedTuple[]

    for config in IFOREST_NOVELTY_CONFIGS
        metrics, predictions = evaluate_iforest(x_train, x_eval, eval_frame, config)
        push!(
            experiment_rows,
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
                runtime_seconds = metrics.runtime_seconds,
            ),
        )
        append!(recall_rows, per_class_recall(eval_frame, predictions, config.name))
    end

    for config in OCSVM_CONFIGS
        metrics, predictions = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        push!(
            experiment_rows,
            (
                algorithm = "one_class_svm",
                config_name = config.name,
                parameters = "kernel=rbf, nu=$(config.nu), gamma=$(config.gamma)",
                precision = metrics.precision,
                recall = metrics.recall,
                f1 = metrics.f1,
                fpr = metrics.fpr,
                tp = metrics.tp,
                fp = metrics.fp,
                tn = metrics.tn,
                fn = metrics.fn,
                runtime_seconds = metrics.runtime_seconds,
            ),
        )
        append!(recall_rows, per_class_recall(eval_frame, predictions, config.name))
    end

    return DataFrame(experiment_rows), DataFrame(recall_rows)
end

function run_trends(x_train::Matrix{Float64}, x_eval::Matrix{Float64}, eval_frame::DataFrame)
    iforest_rows = NamedTuple[]
    ocsvm_nu_rows = NamedTuple[]
    ocsvm_gamma_rows = NamedTuple[]

    for config in IFOREST_THRESHOLD_TREND
        metrics, _ = evaluate_iforest(x_train, x_eval, eval_frame, config)
        push!(
            iforest_rows,
            (
                algorithm = config.algorithm,
                setting = config.setting,
                threshold_percentile = config.threshold_percentile,
                precision = metrics.precision,
                recall = metrics.recall,
                f1 = metrics.f1,
                fpr = metrics.fpr,
            ),
        )
    end

    for config in OCSVM_NU_TREND
        metrics, _ = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        push!(
            ocsvm_nu_rows,
            (
                algorithm = config.algorithm,
                setting = config.setting,
                nu = config.nu,
                precision = metrics.precision,
                recall = metrics.recall,
                f1 = metrics.f1,
                fpr = metrics.fpr,
            ),
        )
    end

    for config in OCSVM_GAMMA_TREND
        metrics, _ = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        push!(
            ocsvm_gamma_rows,
            (
                algorithm = config.algorithm,
                setting = config.setting,
                gamma = config.gamma,
                precision = metrics.precision,
                recall = metrics.recall,
                f1 = metrics.f1,
                fpr = metrics.fpr,
            ),
        )
    end

    return DataFrame(iforest_rows), DataFrame(ocsvm_nu_rows), DataFrame(ocsvm_gamma_rows)
end

function run_ocsvm_refined_grid(x_train::Matrix{Float64}, x_eval::Matrix{Float64}, eval_frame::DataFrame)
    rows = NamedTuple[]

    for config in OCSVM_REFINED_GRID
        metrics, _ = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        push!(
            rows,
            (
                algorithm = config.algorithm,
                setting = config.setting,
                nu = config.nu,
                gamma = config.gamma,
                precision = metrics.precision,
                recall = metrics.recall,
                f1 = metrics.f1,
                fpr = metrics.fpr,
                tp = metrics.tp,
                fp = metrics.fp,
                tn = metrics.tn,
                fn = metrics.fn,
                runtime_seconds = metrics.runtime_seconds,
            ),
        )
    end

    return DataFrame(rows)
end
