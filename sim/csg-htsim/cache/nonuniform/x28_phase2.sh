#!/bin/bash
# x28 phase-2: cold-restrict ramp full nv line + combos + F1 story + robustness
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x28_phase2.log
: > $LOG
BIN=./sim/tests/htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
BASEOPT="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6"

run() { local TAG=$1; shift
  R=$($BIN $BASEOPT "$@" 2>&1 \
      | grep -oE "validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+" \
      | tr '\n' ' ')
  echo "TAG $TAG $R" >> $LOG
}
Z="-chunk_size 33554432 -demand_file $D"
COLD1="-mosaic_ramp_cold 1"
P="-mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"

# --- cold1 / coldE1 missing nv points + combos, 4 nv line ---
for NV in 90 180 360 2400; do
  V="-flex_nv_gbps $NV"
  run "nv${NV}_cold1" $V $Z $COLD1
  run "nv${NV}_coldE1" $V $Z $COLD1 -mosaic_ramp_event
  run "nv${NV}_cold1_rewf" $V $Z $COLD1 -mosaic_donate_rewaterfill
  run "nv${NV}_cold1_P" $V $Z $COLD1 $P
  run "nv${NV}_cold1_e5" $V $Z $COLD1 -mosaic_hop_progress
done
# --- F1+F2 story completion (nv-dependence) ---
for NV in 180 2400; do
  V="-flex_nv_gbps $NV"
  run "nv${NV}_f12_S16_wm6" $V $Z -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 6
  run "nv${NV}_f12S16wm4_rewf" $V $Z -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 4 -mosaic_donate_rewaterfill
done
# --- F1 x F3 coupling ---
run "nv90_f12S16wm6_cold1" -flex_nv_gbps 90 $Z -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 6 $COLD1
run "nv360_f12S16wm6_cold1" -flex_nv_gbps 360 $Z -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 6 $COLD1
run "nv360_f12S16wm4_rewf_cold1" -flex_nv_gbps 360 $Z -mosaic_wave_bytes 16777216 -mosaic_wave_align -mosaic_wave_min 4 -mosaic_donate_rewaterfill $COLD1
# --- W interplay for cold1 @ nv90 ---
for W in 4 8 12; do
  run "nv90_cold1_W$W" -flex_nv_gbps 90 $Z $COLD1 -mosaic_waves $W
done
# --- robustness: cold1 vs base on other demands/scenes ---
for NV in 360 2400; do
  V="-flex_nv_gbps $NV -chunk_size 2097152 -message_size 2097152"
  run "uni${NV}_base" $V
  run "uni${NV}_cold1" $V $COLD1
done
PS=cache/nonuniform/demand_n32_persrc1.0_seed42.txt
run "persrc360_base" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $PS
run "persrc360_cold1" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $PS $COLD1
for SD in 43 44; do
  DS=cache/nonuniform/demand_n32_zipf1.0_seed${SD}.txt
  run "seed${SD}_90_base"  -flex_nv_gbps 90  -chunk_size 33554432 -demand_file $DS
  run "seed${SD}_90_cold1" -flex_nv_gbps 90  -chunk_size 33554432 -demand_file $DS $COLD1
  run "seed${SD}_360_base"  -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $DS
  run "seed${SD}_360_cold1" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $DS $COLD1
done
for A in 1.5 2.0; do
  DA=cache/nonuniform/demand_n32_zipf${A}_seed42.txt
  run "zipf${A}_90_base"  -flex_nv_gbps 90  -chunk_size 67108864 -demand_file $DA
  run "zipf${A}_90_cold1" -flex_nv_gbps 90  -chunk_size 67108864 -demand_file $DA $COLD1
  run "zipf${A}_360_base"  -flex_nv_gbps 360 -chunk_size 67108864 -demand_file $DA
  run "zipf${A}_360_cold1" -flex_nv_gbps 360 -chunk_size 67108864 -demand_file $DA $COLD1
done
KH=cache/nonuniform/demand_n32_khot2_seed42.txt
run "khot2_360_base"  -flex_nv_gbps 360 -chunk_size 16777216 -demand_file $KH
run "khot2_360_cold1" -flex_nv_gbps 360 -chunk_size 16777216 -demand_file $KH $COLD1
run "degr26x4_2400_base"  -flex_nv_gbps 2400 -flex_degrade_gpu 26 -flex_degrade_factor 4 $Z
run "degr26x4_2400_cold1" -flex_nv_gbps 2400 -flex_degrade_gpu 26 -flex_degrade_factor 4 $Z $COLD1
run "taper90_360_base"  -flex_nv_gbps 360 -flex_taper_gbps 90 $Z
run "taper90_360_cold1" -flex_nv_gbps 360 -flex_taper_gbps 90 $Z $COLD1
echo DONE >> $LOG
