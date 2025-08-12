#!/usr/bin/env bash
set -euo pipefail

# Use default GPU (GPU 0). Comment this out to force CPU-only tests
export CUDA_VISIBLE_DEVICES="0"

export PYTHONPATH=src

# Auto-discover and run with explicit summary
python tools/run_unittests.py 2>&1 | tee test.txt