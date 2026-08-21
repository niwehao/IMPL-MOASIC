#!/bin/bash
# 43_ocs_arbiter: MOSAIC-O-A -- the control-plane request arbiter over
# the wave scheme (htsim_optical -wave -wave_arb). n=32, k=1, 800G/500ns,
# 4MB chunks, pay-initial (same conventions as 35/39/40). Opponents are
# NOT rerun (frozen archives: 35 barrier best, 39 PR best, 40 O-W).
#
# Columns:
#   arb         MOSAIC-O-A, canon wave params + arbiter (auto rq gate)
#   arb_rqF     -wave_rq forced on where the auto gate chose fine waves
#   arb_static  lambda frozen at the prior (feedback value under arb)
#   arb_drift   runtime drift absorption (same files/instants as x40)
#   arb_n64     large-R spot cells vs frozen archives
#   arb_ctlD    control round-trip latency D us per activation
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
LOG=${LOG:-cache/optical/x43_arbiter.log}
: > "$LOG"
md5sum $BIN >> "$LOG"

dem_file() {
  case $1 in
    uniform) echo cache/optical/demand_n32_uniform.txt;;
    zipf10)  echo cache/nonuniform/demand_n32_zipf1.0_seed42.txt;;
    zipf20)  echo cache/nonuniform/demand_n32_zipf2.0_seed42.txt;;
    persrc)  echo cache/nonuniform/demand_n32_persrc1.0_seed42.txt;;
  esac
}

run1() { # tag, extra args...
  local TAG=$1; shift
  echo "$TAG $($BIN "$@" 2>/dev/null | tail -1)" >> "$LOG"
}

RS="10 100 1000 10000"

# ---- main grid (canon wave params + arbiter, all-auto)
for DEM in uniform zipf10 zipf20 persrc; do
  DF=$(dem_file $DEM)
  for R in $RS; do
    run1 "arb $DEM $R" -wave -nodes 32 -reconf_us $R -demand_file $DF \
      -wave_arb
  done
done

# ---- ablations: forced rq at the fine-wave cells; static lambda
for CELL in "uniform 10" "zipf10 10" "zipf20 10" "persrc 10"; do
  set -- $CELL
  run1 "arb_rqF $1 $2" -wave -nodes 32 -reconf_us $2 \
    -demand_file $(dem_file $1) -wave_arb -wave_rq
done
for CELL in "uniform 10" "zipf10 10" "persrc 10" "uniform 100"; do
  set -- $CELL
  run1 "arb_static $1 $2" -wave -nodes 32 -reconf_us $2 \
    -demand_file $(dem_file $1) -wave_arb -wave_static
done

# ---- runtime drift (x39/x40 comparison points)
for T in 3000 5000 8000 9500; do
  run1 "arb_drift zipf10 1000 t$T" -wave -nodes 32 -reconf_us 1000 \
    -demand_file $(dem_file zipf10) \
    -drift_file cache/optical/drift_half_seed44.txt -drift_us $T -wave_arb
done
run1 "arb_drift zipf10 100 t3000" -wave -nodes 32 -reconf_us 100 \
  -demand_file $(dem_file zipf10) \
  -drift_file cache/optical/drift_half_seed44.txt -drift_us 3000 -wave_arb

# ---- n64 spot cells (39/40-group frozen archives)
Z64=cache/nonuniform/demand_n64_zipf1.0_seed42.txt
P64=cache/nonuniform/demand_n64_persrc1.0_seed42.txt
run1 "arb_n64 zipf10 100"  -wave -nodes 64 -reconf_us 100  -demand_file $Z64 -wave_arb
run1 "arb_n64 zipf10 1000" -wave -nodes 64 -reconf_us 1000 -demand_file $Z64 -wave_arb
run1 "arb_n64 persrc 1000" -wave -nodes 64 -reconf_us 1000 -demand_file $P64 -wave_arb

# ---- control-plane latency sensitivity (per-activation round trip)
for D in 2 5 20 100; do
  for CELL in "uniform 10" "zipf10 10" "persrc 10" "uniform 100"; do
    set -- $CELL
    run1 "arb_ctl$D $1 $2" -wave -nodes 32 -reconf_us $2 \
      -demand_file $(dem_file $1) -wave_arb -wave_ctl_us $D
  done
done

echo DONE >> "$LOG"
