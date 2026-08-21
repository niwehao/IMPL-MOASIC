#!/bin/bash
# 40_wave_ocs: MOSAIC-O-W -- the EPS wave/lambda scheme ported to the OCS
# fabric (htsim_optical -wave). n=32, k=1, 800G/500ns, 4MB chunks,
# pay-initial (same conventions as 35/39). Opponents are NOT rerun
# (frozen archives: 35_optical_reconfig barrier best, 39_partial_reconf
# PR/async best).
#
# Columns:
#   wave        MOSAIC-O-W, canon params (W6 g0.7 f0.02 t0.75 gate2)
#   waveW1/W3/W12   wave-count ablation
#   wave_static lambda frozen at the prior (feedback value)
#   wave_ident  ring order forced identity (NN auto-gate value)
#   wave_drift  runtime drift absorption (zipf@1000/@100, inject t=3000)
#   n64         zipf/persrc large-R spot cells vs 39-group archive
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
LOG=${LOG:-cache/optical/x40_wave.log}
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

# ---- main grid (canon params) + ablations
for DEM in uniform zipf10 zipf20 persrc; do
  DF=$(dem_file $DEM)
  for R in $RS; do
    run1 "wave $DEM $R"        -wave -nodes 32 -reconf_us $R -demand_file $DF
    run1 "wave_static $DEM $R" -wave -nodes 32 -reconf_us $R -demand_file $DF -wave_static
  done
done

# ---- W ablation (4 representative cells x {1,3,12})
for WW in 1 3 12; do
  for CELL in "uniform 10" "zipf10 10" "zipf10 1000" "persrc 1000"; do
    set -- $CELL
    run1 "waveW$WW $1 $2" -wave -nodes 32 -reconf_us $2 \
      -demand_file $(dem_file $1) -waves $WW
  done
done

# ---- ring-order ablation (identity forced where auto picks NN, and NN
#      forced where auto picks identity)
for R in $RS; do
  run1 "wave_dw0 zipf10 $R" -wave -nodes 32 -reconf_us $R \
    -demand_file $(dem_file zipf10) -wave_dwindow 0
  run1 "wave_nnforce zipf20 $R" -wave -nodes 32 -reconf_us $R \
    -demand_file $(dem_file zipf20) -wave_nn_ring
done

# ---- runtime drift (x39 comparison points: inject 0.5x seed44 at t)
for T in 3000 5000 8000 9500; do
  run1 "wave_drift zipf10 1000 t$T" -wave -nodes 32 -reconf_us 1000 \
    -demand_file $(dem_file zipf10) \
    -drift_file cache/optical/drift_half_seed44.txt -drift_us $T
done
run1 "wave_drift zipf10 100 t3000" -wave -nodes 32 -reconf_us 100 \
  -demand_file $(dem_file zipf10) \
  -drift_file cache/optical/drift_half_seed44.txt -drift_us 3000

# ---- n64 spot cells (39-group archive: zipf@1000 43515.7/20589.6,
#      zipf@100 14245.5/11902, persrc@1000 35737.4/20098.7)
Z64=cache/nonuniform/demand_n64_zipf1.0_seed42.txt
P64=cache/nonuniform/demand_n64_persrc1.0_seed42.txt
run1 "wave_n64 zipf10 100"  -wave -nodes 64 -reconf_us 100  -demand_file $Z64
run1 "wave_n64 zipf10 1000" -wave -nodes 64 -reconf_us 1000 -demand_file $Z64
run1 "wave_n64 persrc 1000" -wave -nodes 64 -reconf_us 1000 -demand_file $P64

echo DONE >> "$LOG"
