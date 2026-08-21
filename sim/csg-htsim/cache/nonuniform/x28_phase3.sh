#!/bin/bash
# x28 phase-3: spine-gate verification + nv crossover (force) + coldE1 combos + robustness
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x28_phase3.log
: > $LOG
BIN=./sim/tests/htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
BASEOPT="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6"
run() { local TAG=$1; shift
  R=$($BIN $BASEOPT "$@" 2>&1 \
      | grep -oE "validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+|mosaic_ramp_cold .*" \
      | tr '\n' ' ')
  echo "TAG $TAG $R" >> $LOG
}
Z="-chunk_size 33554432 -demand_file $D"
CE1="-mosaic_ramp_cold 1 -mosaic_ramp_event"
C1="-mosaic_ramp_cold 1"

# --- flags-off regression after gate patch ---
run "reg_c6_360" -flex_nv_gbps 360 $Z
run "reg_c4_360" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $D -mosaic_corridors 4
# --- gate2 auto shutoff verification (expect bitwise base) ---
run "gate_2400_coldE1" -flex_nv_gbps 2400 $Z $CE1
run "gate_taper90_360_cold1" -flex_nv_gbps 360 -flex_taper_gbps 90 $Z $C1
run "gate_degr26x4_2400_cold1" -flex_nv_gbps 2400 -flex_degrade_gpu 26 -flex_degrade_factor 4 $Z $C1
run "gate_khot2_360_cold1" -flex_nv_gbps 360 -chunk_size 16777216 -demand_file cache/nonuniform/demand_n32_khot2_seed42.txt $C1
# --- gate ON points unchanged ---
for NV in 90 180 360; do
  run "gateon_nv${NV}_coldE1" -flex_nv_gbps $NV $Z $CE1
done
# --- nv crossover with force (420..720) + auto behavior ---
for NV in 420 480 600 720; do
  run "force_nv${NV}_coldE1" -flex_nv_gbps $NV $Z $CE1 -mosaic_ramp_cold_force
  run "auto_nv${NV}_coldE1" -flex_nv_gbps $NV $Z $CE1
done
run "force_nv2400_cold2" -flex_nv_gbps 2400 $Z -mosaic_ramp_cold 2 -mosaic_ramp_cold_force
run "force_nv720_cold2" -flex_nv_gbps 720 $Z -mosaic_ramp_cold 2 -mosaic_ramp_cold_force
# --- coldE1 combos ---
P="-mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
for NV in 90 180 360 2400; do
  run "nv${NV}_coldE1_P" -flex_nv_gbps $NV $Z $CE1 $P
done
run "nv180_coldE1_rewf" -flex_nv_gbps 180 $Z $CE1 -mosaic_donate_rewaterfill
run "nv180_coldE1_skew2" -flex_nv_gbps 180 $Z $CE1 -mosaic_skew_epoch 2
# --- recommended-config robustness (coldE1, auto gate) ---
for SD in 43 44; do
  DS=cache/nonuniform/demand_n32_zipf1.0_seed${SD}.txt
  run "seed${SD}_90_coldE1"  -flex_nv_gbps 90  -chunk_size 33554432 -demand_file $DS $CE1
  run "seed${SD}_360_coldE1" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $DS $CE1
done
for A in 1.5 2.0; do
  DA=cache/nonuniform/demand_n32_zipf${A}_seed42.txt
  run "zipf${A}_90_coldE1"  -flex_nv_gbps 90  -chunk_size 67108864 -demand_file $DA $CE1
  run "zipf${A}_360_coldE1" -flex_nv_gbps 360 -chunk_size 67108864 -demand_file $DA $CE1
done
run "persrc360_coldE1" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt $CE1
run "uni360_coldE1" -flex_nv_gbps 360 -chunk_size 2097152 -message_size 2097152 $CE1
run "uni2400_coldE1" -flex_nv_gbps 2400 -chunk_size 2097152 -message_size 2097152 $CE1
echo DONE >> $LOG
