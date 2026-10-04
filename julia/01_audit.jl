include("common.jl")

function main()
    frame, summary = dataset_audit()
    mkpath(RESULTS_DIR)

    CSV.write(joinpath(RESULTS_DIR, "julia_audit_files.csv"), frame)
    grouped = combine(groupby(frame, :coarse_label), :rows => sum => :rows)
    sort!(grouped, :rows, rev = true)
    CSV.write(joinpath(RESULTS_DIR, "julia_audit_by_label.csv"), grouped)

    summary_frame = DataFrame(item = String[], value = String[])
    for (key, value) in pairs(summary)
        push!(summary_frame, (item = string(key), value = string(value)))
    end
    CSV.write(joinpath(RESULTS_DIR, "julia_audit_summary.csv"), summary_frame)

    println("Zapisano:")
    println(joinpath(RESULTS_DIR, "julia_audit_files.csv"))
    println(joinpath(RESULTS_DIR, "julia_audit_by_label.csv"))
    println(joinpath(RESULTS_DIR, "julia_audit_summary.csv"))
end

main()
