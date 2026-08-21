#!/bin/bash
# One-click standard benchmark for an EPS scheme (2026-08-05, x41 起).
#
#   NAME=myscheme SCHED=mosaic_schedule FLAGS="-mosaic_ledger ..." \
#     bash script/bench_scheme.sh
#
# Runs the CANONICAL CELL SET (mainline 8nv / seeds / alpha / persrc /
# uniform / khot / msg-size / host shapes / n64 / pod grid / unannounced
# degrade / taper / stale-prior drift -- the union of the 21/38/41-group
# suites), writes cache/nonuniform/bench_<NAME>.log, then splices the
# results against the frozen baselines (experiments/mosaic_x1/
# bench_baselines.csv: bound / ECMP / FAST / MOSAIC-U(38) / MOSAIC-U2
# (41 SOTA)) via tools/bench_compare.py ->
#   experiments/mosaic_x1/bench_runs/<NAME>.csv  (+ stdout table)
#
# Promotion gate (CLAUDE.md 守则): a new scheme may claim SOTA only if
# per-cell >= U2 - 5% with no FAILED; breaches must be reported with the
# aggregate-trade argument spelled out (41 组先例).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
NAME=${NAME:?set NAME=<scheme name>}
SCHED=${SCHED:-mosaic_schedule}
FLAGS=${FLAGS:-}
LOG=cache/nonuniform/bench_${NAME}.log
: > "$LOG"
md5sum $BIN >> "$LOG"
echo "# sched $SCHED flags $FLAGS" >> "$LOG"

D32=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
run() { # cell_id, args...
  local CELL=$1; shift
  echo "CELL $CELL $($BIN -packet_size 1024 -workload A2A -all_to_all \
    -store_forward -quiet -scheduler $SCHED $FLAGS "$@" 2>/dev/null \
    | grep -oE 'validation (PASSED|FAILED).*completion_us [0-9.]+|queue_drops [0-9]+' \
    | tr '\n' ' ')" >> "$LOG"
}
RAIL="-nodes 32 -topology flex_rail -flex_nic_gbps 180 -flex_taper_gbps 180 -mosaic_corridors 6"

# ---- mainline (zipf1.0/seed42 x 8 nv)
for NV in 2400 720 600 480 420 360 180 90; do
  run "main_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 -demand_file $D32
done
# ---- seeds / alpha / persrc / uniform
for S in 43 44; do for NV in 2400 360 90; do
  run "seed${S}_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 \
    -demand_file cache/nonuniform/demand_n32_zipf1.0_seed${S}.txt
done; done
for A in 0.5 1.5 2.0; do
  case $A in 0.5) CH=8388608;; *) CH=67108864;; esac
  for NV in 2400 360 90; do
    run "zipf${A}_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size $CH \
      -demand_file cache/nonuniform/demand_n32_zipf${A}_seed42.txt
  done
done
for NV in 2400 360 90; do
  run "persrc_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 \
    -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt
  run "uniform_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 2097152 \
    -message_size 2097152
done
# ---- khot / msg size
for K in 2 4 8; do
  case $K in 8) CH=8388608;; *) CH=16777216;; esac
  for NV in 360 90; do
    run "khot${K}_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size $CH \
      -demand_file cache/nonuniform/demand_n32_khot${K}_seed42.txt
  done
done
for NV in 360 90; do
  run "m512k_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 8388608 \
    -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42_m512k.txt
  run "m8m_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 134217728 \
    -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42_m8m.txt
done
# ---- host shapes
for NV in 2400 360 90; do
  run "shape8x4_nv$NV" -nodes 32 -topology flex_rail -flex_nic_gbps 180 \
    -flex_taper_gbps 180 -mosaic_corridors 6 -flex_hosts 8 \
    -flex_gpus_per_host 4 -flex_nv_gbps $NV -chunk_size 33554432 \
    -demand_file $D32
  run "shape2x16_nv$NV" -nodes 32 -topology flex_rail -flex_nic_gbps 180 \
    -flex_taper_gbps 180 -mosaic_corridors 6 -flex_hosts 2 \
    -flex_gpus_per_host 16 -flex_nv_gbps $NV -chunk_size 33554432 \
    -demand_file $D32
done
# ---- n64
for NV in 2400 360; do
  run "n64_nv$NV" -nodes 64 -topology flex_rail -flex_nic_gbps 180 \
    -flex_taper_gbps 180 -mosaic_corridors 6 -flex_hosts 8 \
    -flex_gpus_per_host 8 -flex_nv_gbps $NV -chunk_size 16777216 \
    -demand_file cache/nonuniform/demand_n64_zipf1.0_seed42.txt
done
# ---- pod grid (C=5)
for P in "2400 720" "360 720" "360 360" "90 720"; do set -- $P
  run "pod_nv$1_up$2" -nodes 32 -topology flex_pod -flex_spines 4 \
    -flex_nic_gbps 180 -flex_taper_gbps $2 -mosaic_corridors 5 \
    -flex_nv_gbps $1 -chunk_size 33554432 -demand_file $D32
done
# ---- unannounced degrade (scheduler not told)
for NV in 2400 360; do for F in 2 4; do
  run "deghot${F}_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 \
    -demand_file $D32 -flex_degrade_gpu 26 -flex_degrade_factor $F
  run "degcold${F}_nv$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 \
    -demand_file $D32 -flex_degrade_gpu 7 -flex_degrade_factor $F
done; done
# ---- taper (uniform + nonuniform)
for NV in 2400 360; do for TP in 90 60; do
  run "utaper_nv${NV}_tp$TP" -nodes 32 -topology flex_rail \
    -flex_nic_gbps 180 -flex_taper_gbps $TP -mosaic_corridors 6 \
    -flex_nv_gbps $NV -chunk_size 2097152 -message_size 2097152
done; done
for TP in 90 360; do
  run "ntaper_nv360_tp$TP" -nodes 32 -topology flex_rail \
    -flex_nic_gbps 180 -flex_taper_gbps $TP -mosaic_corridors 6 \
    -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $D32
done
# ---- stale-prior drift (persrc; fresh/stale must both pass)
run "drift_fresh_nv360" $RAIL -flex_nv_gbps 360 -chunk_size 33554432 \
  -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt
run "drift_stale_nv360" $RAIL -flex_nv_gbps 360 -chunk_size 33554432 \
  -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt \
  -mosaic_prior_demand_file cache/nonuniform/demand_n32_persrc1.0_seed43.txt

echo DONE >> "$LOG"
python3 tools/bench_compare.py "$NAME"
