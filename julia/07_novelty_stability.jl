using CSV
using DataFrames
using Statistics

include("common.jl")
include("novelty_stage.jl")

const STABILITY_SEEDS = [MODEL_RANDOM_SEED, MODEL_RANDOM_SEED + 1, MODEL_RANDOM_SEED + 2]

function main()
    train, eval_frame = load_samples()
    x_train, x_eval, _ = prepare_features(train, eval_frame)

    rows = NamedTuple[]

    for seed in STABILITY_SEEDS
        for config in IFOREST_NOVELTY_CONFIGS
            metrics, _ = evaluate_iforest(x_train, x_eval, eval_frame, config; random_seed = seed)
            push!(
                rows,
                (
                    algorithm = "isolation_forest",
                    config_name = config.name,
                    seed = seed,
                    precision = metrics.precision,
                    recall = metrics.recall,
                    f1 = metrics.f1,
                    fpr = metrics.fpr,
                ),
            )
        end

        for config in OCSVM_CONFIGS
            metrics, _ = evaluate_ocsvm(x_train, x_eval, eval_frame, config; random_seed = seed)
            push!(
                rows,
                (
                    algorithm = "one_class_svm",
                    config_name = config.name,
                    seed = seed,
                    precision = metrics.precision,
                    recall = metrics.recall,
                    f1 = metrics.f1,
                    fpr = metrics.fpr,
                ),
            )
        end
    end

    details = DataFrame(rows)
    sort!(details, [:algorithm, :config_name, :seed])

    grouped = groupby(details, [:algorithm, :config_name])
    summary = combine(
        grouped,
        :precision => mean => :precision_mean,
        :precision => std => :precision_std,
        :recall => mean => :recall_mean,
        :recall => std => :recall_std,
        :f1 => mean => :f1_mean,
        :f1 => std => :f1_std,
        :fpr => mean => :fpr_mean,
        :fpr => std => :fpr_std,
    )
    sort!(summary, [:algorithm, :config_name])

    mkpath(RESULTS_DIR)
    details_path = joinpath(RESULTS_DIR, "julia_novelty_stability_details.csv")
    summary_path = joinpath(RESULTS_DIR, "julia_novelty_stability_summary.csv")
    CSV.write(details_path, details)
    CSV.write(summary_path, summary)

    println("Zapisano:")
    println(details_path)
    println(summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
