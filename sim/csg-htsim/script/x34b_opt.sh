#!/bin/bash
# 34_opt batch 2: pair_color auto-gate verification + main-line regression
# + guard at the M3 damage points (weak nv).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
LOG=${LOG:-cache/nonuniform/x34b_opt.log}
: > "$LOG"
echo "binary_md5 $(md5sum $BIN | cut -d' ' -f1)" >> "$LOG"

run() {
  local tag="$1"; shift
  local out gate
  out=$("$BIN" "$@" 2>&1)
  gate=$(echo "$out" | grep -o "mosaic_pair_color_auto row_cov [0-9.e-]* K [0-9]*" | head -1)
  local v=$(echo "$out" | grep -E "^validation")
  local status=$(echo "$v" | grep -o "PASSED\|FAILED" | head -1)
  local us=$(echo "$v" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
  echo "$tag $status $us [$gate]" >> "$LOG"
  echo "$tag $status $us [$gate]"
}

COMMON="-nodes 32 -topology flex_rail -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
DZ1=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
DZ44=cache/nonuniform/demand_n32_zipf1.0_seed44.txt
DZ15=cache/nonuniform/demand_n32_zipf1.5_seed42.txt
DPS=cache/nonuniform/demand_n32_persrc1.0_seed42.txt
UNI="-chunk_size 2097152 -message_size 2097152"
NZ1="-chunk_size 33554432"
NZ15="-chunk_size 67108864"

# --- auto gate verification (expect K=32 row-balanced, K=0 persrc)
run auto_uniform_2400 $COMMON -flex_nv_gbps 2400 $UNI -mosaic_pair_color_auto
run auto_uniform_360  $COMMON -flex_nv_gbps 360  $UNI -mosaic_pair_color_auto
run auto_persrc_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DPS -mosaic_pair_color_auto
run auto_seed44_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DZ44 -mosaic_pair_color_auto
run auto_zipf15_2400  $COMMON -flex_nv_gbps 2400 $NZ15 -demand_file $DZ15 -mosaic_pair_color_auto

# --- main-line 8 nv regression with auto (vs 21_design_iter mosaic_c6_us)
for NV in 2400 720 600 480 420 360 180 90; do
  run auto_zipf10_$NV $COMMON -flex_nv_gbps $NV $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto
done

# --- guard at M3 damage points
run g05_zipf10_90   $COMMON -flex_nv_gbps 90  $NZ1 -demand_file $DZ1 -mosaic_guard 0.05
run g05_uniform_90  $COMMON -flex_nv_gbps 90  $UNI -mosaic_guard 0.05
run base_uniform_90 $COMMON -flex_nv_gbps 90  $UNI
run g10_zipf10_90   $COMMON -flex_nv_gbps 90  $NZ1 -demand_file $DZ1 -mosaic_guard 0.10

echo DONE >> "$LOG"
echo DONE
