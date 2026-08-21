#!/bin/bash
# x41 promotion gate: paired U-config vs LU-config on every mainline-
# adjacent regime the 38-group unified sweep covered (same binary, same
# flags except -mosaic_ledger). U = C6+PC(auto)+PACE (pod: C5).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
LOG=${LOG:-cache/nonuniform/x41_gate.log}
: > "$LOG"
md5sum $BIN >> "$LOG"

run() { local TAG=$1; shift
  echo "$TAG $($BIN -packet_size 1024 -scheduler mosaic_schedule \
    -workload A2A -all_to_all -store_forward -quiet "$@" 2>/dev/null \
    | grep -oE 'validation (PASSED|FAILED).*completion_us [0-9.]+|queue_drops [0-9]+' \
    | tr '\n' ' ')" >> "$LOG"
}

U="-mosaic_corridors 6 -mosaic_pair_color_auto -mosaic_cold_pace 1"
RAIL="-nodes 32 -topology flex_rail -flex_nic_gbps 180 -flex_taper_gbps 180"

pair() { # tag, chunk, args...
  local TAG=$1 CH=$2; shift 2
  run "U $TAG"  -chunk_size $CH $U "$@"
  run "LU $TAG" -chunk_size $CH $U -mosaic_ledger "$@"
}

# khot family
for K in 2 4 8; do
  case $K in 8) CH=8388608;; *) CH=16777216;; esac
  for NV in 360 90; do
    pair "khot$K nv=$NV" $CH $RAIL -flex_nv_gbps $NV \
      -demand_file cache/nonuniform/demand_n32_khot${K}_seed42.txt
  done
done
# message size 0.25x / 4x
for NV in 360 90; do
  pair "m512k nv=$NV" 8388608 $RAIL -flex_nv_gbps $NV \
    -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42_m512k.txt
  pair "m8m nv=$NV" 134217728 $RAIL -flex_nv_gbps $NV \
    -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42_m8m.txt
done
# host shapes
for NV in 2400 360 90; do
  pair "shape8x4 nv=$NV" 33554432 -nodes 32 -topology flex_rail \
    -flex_nic_gbps 180 -flex_taper_gbps 180 -flex_hosts 8 \
    -flex_gpus_per_host 4 -flex_nv_gbps $NV \
    -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
  pair "shape2x16 nv=$NV" 33554432 -nodes 32 -topology flex_rail \
    -flex_nic_gbps 180 -flex_taper_gbps 180 -flex_hosts 2 \
    -flex_gpus_per_host 16 -flex_nv_gbps $NV \
    -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
done
# n=64 weak scaling
for NV in 2400 360; do
  pair "n64 nv=$NV" 16777216 -nodes 64 -topology flex_rail \
    -flex_nic_gbps 180 -flex_taper_gbps 180 -flex_hosts 8 \
    -flex_gpus_per_host 8 -flex_nv_gbps $NV \
    -demand_file cache/nonuniform/demand_n64_zipf1.0_seed42.txt
done
# pod extra points (4:1 oversub + weak NV)
UPOD="-mosaic_corridors 5 -mosaic_pair_color_auto -mosaic_cold_pace 1"
run "U pod nv=360 up=360"  -chunk_size 33554432 $UPOD -nodes 32 \
  -topology flex_pod -flex_spines 4 -flex_taper_gbps 360 \
  -flex_nic_gbps 180 -flex_nv_gbps 360 \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
run "LU pod nv=360 up=360" -chunk_size 33554432 $UPOD -mosaic_ledger \
  -nodes 32 -topology flex_pod -flex_spines 4 -flex_taper_gbps 360 \
  -flex_nic_gbps 180 -flex_nv_gbps 360 \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
run "U pod nv=90 up=720"  -chunk_size 33554432 $UPOD -nodes 32 \
  -topology flex_pod -flex_spines 4 -flex_taper_gbps 720 \
  -flex_nic_gbps 180 -flex_nv_gbps 90 \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
run "LU pod nv=90 up=720" -chunk_size 33554432 $UPOD -mosaic_ledger \
  -nodes 32 -topology flex_pod -flex_spines 4 -flex_taper_gbps 720 \
  -flex_nic_gbps 180 -flex_nv_gbps 90 \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# LU seed robustness
for S in 43 44; do
  for NV in 2400 360 90; do
    pair "seed$S nv=$NV" 33554432 $RAIL -flex_nv_gbps $NV \
      -demand_file cache/nonuniform/demand_n32_zipf1.0_seed${S}.txt
  done
done
# @90 remediation probe: ledger + PACE without PC
run "L_noPC seed42 nv=90" -chunk_size 33554432 -mosaic_corridors 6 \
  -mosaic_cold_pace 1 -mosaic_ledger $RAIL -flex_nv_gbps 90 \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt

echo DONE >> "$LOG"
