using CSV
using CodecZlib
using DataFrames
using DataStructures
using SHA
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(PROJECT_ROOT, "IoT-DIAD", "Anomaly Detection - Flow based features")
const SAMPLE_DIR = joinpath(PROJECT_ROOT, "samples")
const RESULTS_DIR = joinpath(PROJECT_ROOT, "results")

const TRAIN_SAMPLE_PATH = joinpath(SAMPLE_DIR, "train_benign_30000.csv.gz")
const EVAL_SAMPLE_PATH = joinpath(SAMPLE_DIR, "eval_mixed_12000_14000.csv.gz")

const BENIGN_ROWS = 30_000
const EVAL_BENIGN_ROWS = 12_000
const EVAL_ATTACK_ROWS = 14_000
const TRAIN_BENIGN_SEED = 2026
const EVAL_BENIGN_SEED = 2027
const EVAL_ATTACK_SEED = 2028
const MODEL_RANDOM_SEED = 2029

const HELPER_COLUMNS = ["coarse_label", "fine_label", "binary_label", "source_file", "source_row"]

const EXCLUDED_COLUMNS = Set([
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "coarse_label",
    "fine_label",
    "binary_label",
    "source_file",
    "source_row",
])

struct ScoredRow
    score::UInt64
    row::Vector{String}
end

Base.isless(left::ScoredRow, right::ScoredRow) = isless(left.score, right.score)

function csv_paths(root::String)
    paths = String[]
    for (current_root, _, files) in walkdir(root)
        for file in files
            endswith(file, ".csv") || continue
            push!(paths, joinpath(current_root, file))
        end
    end
    return sort(paths)
end

function first_csv_row(path::String)
    open(path, "r") do io
        eof(io) && return String[]
        line = readline(io)
        isempty(line) && return String[]
        return String.(split(chomp(line), ','))
    end
end

function canonical_header(root::String = DATA_DIR)
    for path in csv_paths(root)
        row = first_csv_row(path)
        if !isempty(row) && row[1] == "Flow ID"
            return row
        end
    end
    error("Nie znaleziono poprawnego naglowka CSV.")
end

function infer_labels(path::String, root::String = DATA_DIR)
    relative = relpath(path, root)
    parts = splitpath(relative)
    coarse = parts[1]
    fine = length(parts) > 2 ? parts[2] : coarse
    return coarse, fine
end

function file_list(root::String = DATA_DIR)
    header = canonical_header(root)
    files = NamedTuple[]
    for path in csv_paths(root)
        first_row = first_csv_row(path)
        coarse, fine = infer_labels(path, root)
        push!(files, (path = path, coarse_label = coarse, fine_label = fine, header_ok = first_row == header))
    end
    return files
end

function each_data_row(callback, path::String, header_ok::Bool, expected_width::Int)
    open(path, "r") do io
        first_line = true
        row_index = 0
        for line in eachline(io)
            if first_line && header_ok
                first_line = false
                continue
            end
            first_line = false

            values = String.(split(chomp(line), ','))
            length(values) == expected_width || continue
            row_index += 1
            callback(row_index, values)
        end
    end
end

function count_rows(path::String, header_ok::Bool, expected_width::Int)
    count_ref = Ref(0)
    each_data_row(path, header_ok, expected_width) do _, _
        count_ref[] += 1
    end
    return count_ref[]
end

function dataset_audit(root::String = DATA_DIR)
    header = canonical_header(root)
    rows = NamedTuple[]
    total_rows = 0
    benign_rows = 0

    for item in file_list(root)
        row_count = count_rows(item.path, item.header_ok, length(header))
        push!(
            rows,
            (
                file = relpath(item.path, root),
                coarse_label = item.coarse_label,
                fine_label = item.fine_label,
                rows = row_count,
                header_ok = item.header_ok,
            ),
        )
        total_rows += row_count
        if item.coarse_label == "Benign"
            benign_rows += row_count
        end
    end

    frame = DataFrame(rows)
    summary = (
        csv_files = nrow(frame),
        rows_total = total_rows,
        benign_rows = benign_rows,
        attack_rows = total_rows - benign_rows,
        bad_headers = sum(.!frame.header_ok),
    )
    return frame, summary
end

function deterministic_score(
    label::String,
    source_file::String,
    source_row::Int,
    row::Vector{String};
    seed::Int = TRAIN_BENIGN_SEED,
)
    message = string(seed, '\x1e', label, '\x1e', source_file, '\x1e', source_row, '\x1e', join(row, '\x1f'))
    digest = sha1(message)
    score = zero(UInt64)
    for byte in digest[1:8]
        score = (score << 8) | UInt64(byte)
    end
    return score
end

function build_attack_targets()
    per_attack = div(EVAL_ATTACK_ROWS, 7)
    return Dict(
        "BruteForce" => per_attack,
        "DDoS" => per_attack,
        "DoS" => per_attack,
        "Mirai" => per_attack,
        "Recon" => per_attack,
        "Spoofing" => per_attack,
        "Web-Based" => EVAL_ATTACK_ROWS - 6 * per_attack,
    )
end

function collect_top_rows(
    targets::Dict{String, Int};
    root::String = DATA_DIR,
    seed::Int = TRAIN_BENIGN_SEED,
    exclude_keys::Set{Tuple{String, Int}} = Set{Tuple{String, Int}}(),
)
    header = canonical_header(root)
    heaps = Dict(label => BinaryMaxHeap{ScoredRow}() for label in keys(targets))

    for item in file_list(root)
        target_size = get(targets, item.coarse_label, 0)
        target_size == 0 && continue

        relative_path = replace(relpath(item.path, root), '\\' => '/')

        each_data_row(item.path, item.header_ok, length(header)) do row_index, values
            key = (relative_path, row_index)
            if key in exclude_keys
                return nothing
            end

            score = deterministic_score(
                item.coarse_label,
                relative_path,
                row_index,
                values;
                seed = seed,
            )
            enriched = vcat(
                values,
                [
                    item.coarse_label,
                    item.fine_label,
                    item.coarse_label == "Benign" ? "0" : "1",
                    relative_path,
                    string(row_index),
                ],
            )
            scored = ScoredRow(score, enriched)
            heap = heaps[item.coarse_label]

            if length(heap) < target_size
                push!(heap, scored)
            elseif score < first(heap).score
                pop!(heap)
                push!(heap, scored)
            end
        end
    end

    columns = vcat(header, HELPER_COLUMNS)
    label_order = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-Based"]
    selected_rows = Vector{Vector{String}}()

    for label in label_order
        haskey(targets, label) || continue
        current_rows = ScoredRow[]
        while !isempty(heaps[label])
            push!(current_rows, pop!(heaps[label]))
        end
        sort!(current_rows; by = item -> item.score)
        append!(selected_rows, [item.row for item in current_rows])
    end

    frame = DataFrame()
    for (index, name) in enumerate(columns)
        frame[!, name] = [row[index] for row in selected_rows]
    end
    return frame
end

function build_sample_frames(; root::String = DATA_DIR)
    train = collect_top_rows(Dict("Benign" => BENIGN_ROWS); root = root, seed = TRAIN_BENIGN_SEED)
    sort!(train, [:source_file, :source_row])

    train_keys = Set((String(row.source_file), parse(Int, String(row.source_row))) for row in eachrow(train))
    eval_benign = collect_top_rows(
        Dict("Benign" => EVAL_BENIGN_ROWS);
        root = root,
        seed = EVAL_BENIGN_SEED,
        exclude_keys = train_keys,
    )
    sort!(eval_benign, [:source_file, :source_row])

    attack_frame = collect_top_rows(build_attack_targets(); root = root, seed = EVAL_ATTACK_SEED)
    sort!(attack_frame, [:coarse_label, :source_file, :source_row])

    eval_frame = vcat(eval_benign, attack_frame)
    return train, eval_frame
end

function write_gzip_csv(path::String, frame::DataFrame)
    open(path, "w") do io
        gz = GzipCompressorStream(io)
        CSV.write(gz, frame)
        close(gz)
    end
end

function read_gzip_csv(path::String)
    open(path, "r") do io
        gz = GzipDecompressorStream(io)
        return CSV.read(gz, DataFrame)
    end
end

function ensure_sample_files(; force::Bool = false)
    mkpath(SAMPLE_DIR)

    if force || !isfile(TRAIN_SAMPLE_PATH) || !isfile(EVAL_SAMPLE_PATH)
        train, eval_frame = build_sample_frames()
        write_gzip_csv(TRAIN_SAMPLE_PATH, train)
        write_gzip_csv(EVAL_SAMPLE_PATH, eval_frame)
    end

    return TRAIN_SAMPLE_PATH, EVAL_SAMPLE_PATH
end

function load_samples()
    ensure_sample_files()
    train = read_gzip_csv(TRAIN_SAMPLE_PATH)
    eval_frame = read_gzip_csv(EVAL_SAMPLE_PATH)
    return train, eval_frame
end

function sample_summary(train::DataFrame, eval_frame::DataFrame)
    train_keys = unique(train[:, [:source_file, :source_row]])
    eval_benign = eval_frame[Int.(eval_frame.binary_label) .== 0, :]
    eval_keys = unique(eval_benign[:, [:source_file, :source_row]])
    overlap = innerjoin(train_keys, eval_keys; on = [:source_file, :source_row])

    attack_frame = eval_frame[Int.(eval_frame.binary_label) .== 1, :]
    counts = combine(groupby(attack_frame, :coarse_label), nrow => :count)
    sort!(counts, :coarse_label)

    rows = NamedTuple[
        (item = "train_benign_rows", value = string(nrow(train))),
        (item = "eval_benign_rows", value = string(nrow(eval_benign))),
        (item = "eval_attack_rows", value = string(nrow(attack_frame))),
        (item = "train_eval_benign_overlap", value = string(nrow(overlap))),
    ]

    for row in eachrow(counts)
        push!(rows, (item = "eval_attack_$(row.coarse_label)", value = string(row.count)))
    end

    return DataFrame(rows)
end

function clean_numeric_column(column)
    out = Vector{Union{Missing, Float64}}(undef, length(column))
    @inbounds for index in eachindex(column)
        value = column[index]
        if ismissing(value)
            out[index] = missing
            continue
        end
        parsed = try
            Float64(value)
        catch
            missing
        end
        if ismissing(parsed) || !isfinite(parsed)
            out[index] = missing
        else
            out[index] = parsed
        end
    end
    return out
end

function prepare_features(train::DataFrame, eval_frame::DataFrame)
    feature_names = String[]
    for name in names(train)
        name_str = String(name)
        name_str in EXCLUDED_COLUMNS && continue
        push!(feature_names, name_str)
    end

    keep = String[]
    medians = Float64[]
    means = Float64[]
    scales = Float64[]

    for name in feature_names
        values = clean_numeric_column(train[!, name])
        observed = collect(skipmissing(values))
        isempty(observed) && continue
        length(unique(observed)) <= 1 && continue

        med = median(observed)
        filled = Vector{Float64}(undef, length(values))
        @inbounds for index in eachindex(values)
            filled[index] = ismissing(values[index]) ? med : values[index]
        end

        mean_value = mean(filled)
        std_value = std(filled; corrected = false)
        std_value > 0 || continue

        push!(keep, name)
        push!(medians, med)
        push!(means, mean_value)
        push!(scales, std_value)
    end

    function transform(frame::DataFrame)
        matrix = Matrix{Float64}(undef, nrow(frame), length(keep))
        for (column_index, name) in enumerate(keep)
            values = clean_numeric_column(frame[!, name])
            med = medians[column_index]
            mu = means[column_index]
            sigma = scales[column_index]
            @inbounds for row_index in 1:nrow(frame)
                value = ismissing(values[row_index]) ? med : values[row_index]
                matrix[row_index, column_index] = (value - mu) / sigma
            end
        end
        return matrix
    end

    return transform(train), transform(eval_frame), keep
end

function preprocessing_summary(train::DataFrame, eval_frame::DataFrame)
    raw_feature_columns = String[]
    for name in names(train)
        name_str = String(name)
        name_str in EXCLUDED_COLUMNS && continue
        push!(raw_feature_columns, name_str)
    end

    kept_columns = String[]
    for name in raw_feature_columns
        values = clean_numeric_column(train[!, name])
        observed = collect(skipmissing(values))
        isempty(observed) && continue
        length(unique(observed)) <= 1 && continue
        push!(kept_columns, name)
    end

    dropped_columns = [name for name in raw_feature_columns if !(name in kept_columns)]
    rows = [
        (item = "train_rows", value = string(nrow(train))),
        (item = "eval_rows", value = string(nrow(eval_frame))),
        (item = "raw_feature_count", value = string(length(raw_feature_columns))),
        (item = "kept_feature_count", value = string(length(kept_columns))),
        (item = "dropped_feature_count", value = string(length(dropped_columns))),
        (item = "excluded_columns", value = join(sort(collect(EXCLUDED_COLUMNS)), ", ")),
        (item = "kept_columns", value = join(kept_columns, ", ")),
        (item = "dropped_columns", value = join(dropped_columns, ", ")),
    ]
    return DataFrame(rows)
end

function metrics_table(eval_frame::DataFrame, predictions::Vector{Int})
    truth = Int.(eval_frame.binary_label)
    tn = sum((truth .== 0) .& (predictions .== 0))
    fp = sum((truth .== 0) .& (predictions .== 1))
    fn = sum((truth .== 1) .& (predictions .== 0))
    tp = sum((truth .== 1) .& (predictions .== 1))

    precision = (tp + fp) == 0 ? 0.0 : tp / (tp + fp)
    recall = (tp + fn) == 0 ? 0.0 : tp / (tp + fn)
    f1 = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
    fpr = (fp + tn) == 0 ? 0.0 : fp / (fp + tn)

    return (
        precision = precision,
        recall = recall,
        f1 = f1,
        fpr = fpr,
        tp = tp,
        fp = fp,
        tn = tn,
        fn = fn,
    )
end
