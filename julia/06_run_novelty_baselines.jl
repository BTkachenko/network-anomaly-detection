using CSV
using DataFrames

include("common.jl")
include("novelty_stage.jl")

function main()
    train, eval_frame = load_samples()
    x_train, x_eval, used_columns = prepare_features(train, eval_frame)

    experiments, recall_by_class = run_final_configs(x_train, x_eval, eval_frame)
    iforest_trend, ocsvm_nu_trend, ocsvm_gamma_trend = run_trends(x_train, x_eval, eval_frame)
    ocsvm_refined_grid = run_ocsvm_refined_grid(x_train, x_eval, eval_frame)

    insertcols!(experiments, 1, :feature_count => fill(length(used_columns), nrow(experiments)))

    mkpath(RESULTS_DIR)
    CSV.write(joinpath(RESULTS_DIR, "julia_novelty_experiments.csv"), experiments)
    CSV.write(joinpath(RESULTS_DIR, "julia_novelty_recall_by_class.csv"), recall_by_class)
    CSV.write(joinpath(RESULTS_DIR, "julia_novelty_iforest_threshold_trend.csv"), iforest_trend)
    CSV.write(joinpath(RESULTS_DIR, "julia_novelty_ocsvm_nu_trend.csv"), ocsvm_nu_trend)
    CSV.write(joinpath(RESULTS_DIR, "julia_novelty_ocsvm_gamma_trend.csv"), ocsvm_gamma_trend)
    CSV.write(joinpath(RESULTS_DIR, "julia_novelty_ocsvm_refined_grid.csv"), ocsvm_refined_grid)

    sort!(experiments, [:algorithm, :f1], rev = [false, true])
    println(experiments)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
