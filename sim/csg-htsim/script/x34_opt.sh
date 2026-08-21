#!/bin/bash
# 34_opt: O1 pair-color + O3 lambda-guard exploration batch.
# One line per run: tag then completion_us (only PASSED runs print a number).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
LOG=${LOG:-cache/nonuniform/x34_opt.log}
: > "$LOG"
echo "binary_md5 $(md5sum $BIN | cut -d' ' -f1)" >> "$LOG"

run() {
  local tag="$1"; shift
  local out
  out=$("$BIN" "$@" 2>&1 | grep -E "^validation")
  local status=$(echo "$out" | grep -o "PASSED\|FAILED" | head -1)
  local us=$(echo "$out" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
  local drops=$(echo "$out" | sed -n 's/.*queue_drops \([0-9]*\).*/\1/p')
  echo "$tag $status $us drops=$drops" >> "$LOG"
  echo "$tag $status $us drops=$drops"
}

COMMON="-nodes 32 -topology flex_rail -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
DZ1=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
DZ44=cache/nonuniform/demand_n32_zipf1.0_seed44.txt
DZ15=cache/nonuniform/demand_n32_zipf1.5_seed42.txt
DPS=cache/nonuniform/demand_n32_persrc1.0_seed42.txt
UNI="-chunk_size 2097152 -message_size 2097152"
NZ1="-chunk_size 33554432"
NZ15="-chunk_size 67108864"

# --- baselines (re-run under this binary; must match archive within noise)
run base_uniform_2400   $COMMON -flex_nv_gbps 2400 $UNI
run base_uniform_360    $COMMON -flex_nv_gbps 360  $UNI
run base_persrc_2400    $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DPS
run base_seed44_2400    $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DZ44
run base_zipf15_2400    $COMMON -flex_nv_gbps 2400 $NZ15 -demand_file $DZ15
run base_zipf10_360     $COMMON -flex_nv_gbps 360  $NZ1 -demand_file $DZ1
run base_zipf10_2400    $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DZ1

# --- O1 pair coloring, K=32 (full diagonal) and K=8
for K in 32 8; do
  run pc${K}_uniform_2400 $COMMON -flex_nv_gbps 2400 $UNI -mosaic_pair_color $K
  run pc${K}_uniform_360  $COMMON -flex_nv_gbps 360  $UNI -mosaic_pair_color $K
  run pc${K}_persrc_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DPS -mosaic_pair_color $K
  run pc${K}_seed44_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DZ44 -mosaic_pair_color $K
  run pc${K}_zipf15_2400  $COMMON -flex_nv_gbps 2400 $NZ15 -demand_file $DZ15 -mosaic_pair_color $K
  run pc${K}_zipf10_360   $COMMON -flex_nv_gbps 360  $NZ1 -demand_file $DZ1 -mosaic_pair_color $K
  run pc${K}_zipf10_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DZ1 -mosaic_pair_color $K
done

# --- O3 lambda guard, eps=0.05
for E in 0.05; do
  run g${E}_uniform_2400 $COMMON -flex_nv_gbps 2400 $UNI -mosaic_guard $E
  run g${E}_uniform_360  $COMMON -flex_nv_gbps 360  $UNI -mosaic_guard $E
  run g${E}_persrc_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DPS -mosaic_guard $E
  run g${E}_zipf10_360   $COMMON -flex_nv_gbps 360  $NZ1 -demand_file $DZ1 -mosaic_guard $E
  run g${E}_zipf10_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DZ1 -mosaic_guard $E
done

echo DONE >> "$LOG"
echo DONE
