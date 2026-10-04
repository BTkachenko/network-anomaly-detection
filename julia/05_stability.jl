using CSV
using DataFrames
using Statistics

include("common.jl")
include("simple_iforest.jl")
include("04_run_models.jl")

const STABILITY_SEEDS = [2027, 2028, 2029]

function main()
    train, eval_frame = load_samples()
    x_train, x_eval, _ = prepare_features(train, eval_frame)

    rows = NamedTuple[]
    for seed in STABILITY_SEEDS
        append!(rows, run_kmeans(x_train, x_eval, eval_frame; random_seed = seed))
        append!(rows, run_iforest(x_train, x_eval, eval_frame; random_seed = seed))
    end

    details = DataFrame(rows)
    insertcols!(details, 3, :seed => repeat(STABILITY_SEEDS, inner = length(KMEANS_CONFIGS) + length(IFOREST_CONFIGS)))
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
    details_path = joinpath(RESULTS_DIR, "julia_stability_details.csv")
    summary_path = joinpath(RESULTS_DIR, "julia_stability_summary.csv")
    CSV.write(details_path, details)
    CSV.write(summary_path, summary)
    println("Zapisano:")
    println(details_path)
    println(summary_path)
end

main()
