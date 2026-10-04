include("common.jl")

function main()
    train, eval_frame = load_samples()
    summary = preprocessing_summary(train, eval_frame)
    mkpath(RESULTS_DIR)
    out_path = joinpath(RESULTS_DIR, "julia_preprocessing_summary.csv")
    CSV.write(out_path, summary)
    println("Zapisano:")
    println(out_path)
end

main()
