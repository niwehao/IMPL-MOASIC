#!/bin/bash
# 36_eps: weak-scaling EPS grid, n in {256,512,1024} (hosts x 8),
# per-src total bytes held at the 32-node contract (65 MB).
# Schemes: mosaic C6, mosaic C6+PC(auto), fast(+guard), static_ecmp.
# Records completion_us, per-A2A plan construction plan_us, wall seconds.
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
LOG=${LOG:-cache/nonuniform/x36_eps.log}
: > "$LOG"
echo "binary_md5 $(md5sum $BIN | cut -d' ' -f1)" >> "$LOG"

run() {
  local tag="$1"; shift
  local t0=$(date +%s.%N)
  local out
  out=$("$BIN" "$@" 2>&1)
  local t1=$(date +%s.%N)
  local v=$(echo "$out" | grep -E "^validation" | head -1)
  local status=$(echo "$v" | grep -o "PASSED\|FAILED" | head -1)
  local us=$(echo "$v" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
  local drops=$(echo "$v" | sed -n 's/.*queue_drops \([0-9]*\).*/\1/p')
  local plan=$(echo "$out" | grep -oE "(mosaic_plan|fast_plan).* plan_us [0-9.]+" | grep -oE "plan_us [0-9.]+" | cut -d' ' -f2)
  local wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  echo "$tag $status $us drops=$drops plan_us=${plan:-0} wall_s=$wall" >> "$LOG"
  echo "$tag $status $us plan_us=${plan:-0} wall_s=$wall"
}

for N in 256 512 1024; do
  H=$((N/8))
  D=cache/nonuniform/demand_n${N}_zipf1.0_seed42.txt
  C="-nodes $N -topology flex_rail -flex_hosts $H -flex_gpus_per_host 8 \
     -flex_nv_gbps 360 -packet_size 1024 -chunk_size 16777216 \
     -workload A2A -all_to_all -store_forward -quiet -demand_file $D"
  run n${N}_mosaic_c6   $C -scheduler mosaic_schedule -mosaic_corridors 6
  run n${N}_mosaic_pc   $C -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_pair_color_auto
  run n${N}_fast        $C -scheduler fast_schedule -fast_alpha_beta_guard
  run n${N}_ecmp        $C -scheduler static_ecmp
done
echo DONE >> "$LOG"
echo DONE
