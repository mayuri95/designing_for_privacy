#!/usr/bin/env bash
set -euo pipefail

core=0

for a in {0..9}; do
  for b in {0..1}; do
    for c in {0..2}; do
      echo "Launching on core $core: python3 main.py $a $b $c"

      taskset -c "$core" \
        env OMP_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            VECLIB_MAXIMUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 \
            TORCH_NUM_THREADS=1 \
        uv run python3 -u "$a" "$b" "$c" &

      core=$((core + 1))
    done
  done
done

wait
echo "All 60 jobs finished."