#!/bin/bash
# 34_opt batch 3: diag-CoV gate verification + pair_color x SOTA-config
# composition (PACE / R+PACE).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
LOG=${LOG:-cache/nonuniform/x34c_opt.log}
: > "$LOG"
echo "binary_md5 $(md5sum $BIN | cut -d' ' -f1)" >> "$LOG"

run() {
  local tag="$1"; shift
  local out gate
  out=$("$BIN" "$@" 2>&1)
  gate=$(echo "$out" | grep -o "mosaic_pair_color_auto diag_cov [0-9.e-]* K [0-9]*" | head -1)
  local v=$(echo "$out" | grep -E "^validation")
  local status=$(echo "$v" | grep -o "PASSED\|FAILED" | head -1)
  local us=$(echo "$v" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
  echo "$tag $status $us [$gate]" >> "$LOG"
  echo "$tag $status $us [$gate]"
}

COMMON="-nodes 32 -topology flex_rail -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
DZ1=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
DPS=cache/nonuniform/demand_n32_persrc1.0_seed42.txt
UNI="-chunk_size 2097152 -message_size 2097152"
NZ1="-chunk_size 33554432"
R="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
PACE="-mosaic_cold_pace 1"

# --- gate verification (expect: uniform/zipf K=32, persrc K=0)
run gate_uniform_2400 $COMMON -flex_nv_gbps 2400 $UNI -mosaic_pair_color_auto
run gate_persrc_2400  $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DPS -mosaic_pair_color_auto
run gate_zipf10_360   $COMMON -flex_nv_gbps 360  $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto

# --- composition with SOTA configs on the main line
# SOTA refs: 90 C6+R+PACE 15567 | 180 C6+R+PACE 11727 | 360 C6+PACE 7917
#            420 C6+R+PACE 7180 | 480 C6+PACE 6370 | 600 C6+PACE 5480
#            720 C6+PACE 4885 | 2400 C6+R 3405
run pcP_zipf10_360  $COMMON -flex_nv_gbps 360  $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $PACE
run pcP_zipf10_480  $COMMON -flex_nv_gbps 480  $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $PACE
run pcP_zipf10_600  $COMMON -flex_nv_gbps 600  $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $PACE
run pcP_zipf10_720  $COMMON -flex_nv_gbps 720  $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $PACE
run pcRP_zipf10_90  $COMMON -flex_nv_gbps 90   $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $R $PACE
run pcRP_zipf10_180 $COMMON -flex_nv_gbps 180  $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $R $PACE
run pcRP_zipf10_420 $COMMON -flex_nv_gbps 420  $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $R $PACE
run pcR_zipf10_2400 $COMMON -flex_nv_gbps 2400 $NZ1 -demand_file $DZ1 -mosaic_pair_color_auto $R

echo DONE >> "$LOG"
echo DONE
