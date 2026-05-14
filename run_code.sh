#!/bin/bash
set -e

sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj

cd "$(dirname "$0")/src"
echo "Now at: $(pwd)"

/home/js/miniconda3/bin/conda run -n python_cource python train_mlp.py
