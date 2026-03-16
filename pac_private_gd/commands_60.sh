#!/usr/bin/env bash
set -euo pipefail

source ../.venv/bin/activate

core=0

for a in {0..4}; do
  for b in {0..1}; do
    for c in {0..4}; do
      echo "Launching on core $core: python3 main.py $a $b $c"

      taskset -c "$core" \
        env OMP_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            VECLIB_MAXIMUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 \
            TORCH_NUM_THREADS=1 \
        python3 main.py "$a" "$b" "$c" &

      core=$((core + 1))
    done
  done
done

wait
echo "All 50 jobs finished."