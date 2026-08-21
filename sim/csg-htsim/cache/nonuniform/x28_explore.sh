#!/bin/bash
# x28 phase-1 exploration: F1 fixed wave SIZE / F2 deadline-aligned epoch /
# F3 corridor ramp — solo grids + combos with MOSAIC-P parts, nv90+nv360.
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x28_explore.log
: > $LOG
BIN=./sim/tests/htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
NU="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -chunk_size 33554432 -topology flex_rail -demand_file $D -scheduler mosaic_schedule -mosaic_corridors 6"

run() { local TAG=$1; shift
  R=$($BIN $NU "$@" 2>&1 \
      | grep -oE "validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+" \
      | tr '\n' ' ')
  echo "TAG $TAG $R" >> $LOG
}

for NV in 90 360; do
  V="-flex_nv_gbps $NV"
  # --- F1 wave-size sweep, no align ---
  for S in 8388608 16777216 33554432 67108864; do
    run "nv${NV}_f1_S${S}" $V -mosaic_wave_bytes $S
  done
  # --- F1+F2 align ---
  for S in 8388608 16777216 33554432 67108864; do
    run "nv${NV}_f12_S${S}" $V -mosaic_wave_bytes $S -mosaic_wave_align
  done
  # --- F1+F2 with wave_min floor (cold keeps adaptivity) ---
  for WM in 2 4 6; do
    run "nv${NV}_f12_S16_wm${WM}" $V -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min $WM
  done
  # --- F3 all-dst ramp, wave-driven / event-driven ---
  for K in 1 2 3; do
    run "nv${NV}_ramp${K}" $V -mosaic_ramp_start $K
    run "nv${NV}_rampE${K}" $V -mosaic_ramp_start $K -mosaic_ramp_event
  done
  # --- F3b cold-restrict ramp ---
  for K in 1 2 3; do
    run "nv${NV}_cold${K}" $V -mosaic_ramp_cold $K
    run "nv${NV}_coldE${K}" $V -mosaic_ramp_cold $K -mosaic_ramp_event
  done
  # --- combos with MOSAIC-P ingredients ---
  P="-mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
  run "nv${NV}_P" $V $P
  run "nv${NV}_P_cold2" $V $P -mosaic_ramp_cold 2
  run "nv${NV}_P_coldE2" $V $P -mosaic_ramp_cold 2 -mosaic_ramp_event
  run "nv${NV}_P_ramp2" $V $P -mosaic_ramp_start 2
  # F1+F2 replacing E3 (align is the adaptive skew precedence), + E2
  run "nv${NV}_f12S16_rewf" $V -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_donate_rewaterfill
  run "nv${NV}_f12S16wm4_rewf" $V -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 4 -mosaic_donate_rewaterfill
  run "nv${NV}_f12S32wm4_rewf" $V -mosaic_wave_bytes 33554432 -mosaic_wave_align -mosaic_wave_min 4 -mosaic_donate_rewaterfill
  # F1+F2 stacked ON TOP of E3 (offsets add)
  run "nv${NV}_f12S16wm4_P" $V -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 4 $P
  # everything: wave-size+align+rewf+cold-ramp
  run "nv${NV}_f12S16wm4_rewf_cold2" $V -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 4 -mosaic_donate_rewaterfill -mosaic_ramp_cold 2
done
echo DONE >> $LOG
