# Julia checkpoint

This directory preserves the earlier Julia implementation for comparison with
the first Python stages. It is not feature-equivalent to the later Python
experiments.

From the repository root, with the dataset in the default location described
in the main README:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/01_audit.jl
julia --project=julia julia/02_prepare_sample.jl
julia --project=julia julia/03_preprocess.jl
julia -t auto --project=julia julia/04_run_models.jl
```

The `NETWORK_ANOMALY_DATA_DIR` override is available in the Python implementation
only. Python and Julia use the shared `samples/` and `results/` directories, with
different prefixes for their result files. The CI job verifies Python only.
