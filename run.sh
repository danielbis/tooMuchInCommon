#!/bin/bash

# When conda activate errors out requesting conda init to be run,
# the eval expression here makes it work without conda init
eval "$(conda shell.bash hook)"
conda activate tmic

mkdir -p experiments/isotropy
mkdir -p experiments/benchmarks
mkdir -p experiments/plots

python ./src/isotropy.py
python ./src/embeddings_benchmark.py
python ./src/vis.py