include("common.jl")

function main()
    force = any(argument == "--force" for argument in ARGS)
    train_path, eval_path = ensure_sample_files(; force = force)
    train, eval_frame = load_samples()
    mkpath(RESULTS_DIR)
    summary_path = joinpath(RESULTS_DIR, "julia_sample_summary.csv")
    CSV.write(summary_path, sample_summary(train, eval_frame))
    println("Zapisano probki:")
    println(train_path)
    println(eval_path)
    println(summary_path)
end

main()
