# Local sample cache

Generated sampled traffic records are intentionally excluded from Git.
Download the dataset and run `python python/02_prepare_sample.py` to create
the benign training and mixed evaluation samples here.

Tests use small synthetic frames constructed in memory. They do not require
or redistribute any dataset records.
