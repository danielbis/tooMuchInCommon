#!/bin/bash

mkdir -p experiments/isotropy
mkdir -p experiments/benchmarks
mkdir -p experiments/plots

python src/isotropy.py

python src/embedding_benchmarks.py

python src/vis.py