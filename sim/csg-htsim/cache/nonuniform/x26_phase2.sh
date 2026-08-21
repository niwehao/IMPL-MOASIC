#!/bin/bash
# x26 phase-2: E5 (hop-progress censoring fix) combos + winner robustness
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x26_phase2.log
: > $LOG
BIN=./sim/tests/htsim_smart_regular
COMMON="-nodes 32 -topology flex_rail -packet_size 1024 -scheduler mosaic_schedule -mosaic_corridors 6 -workload A2A -all_to_all -store_forward -quiet"
D10=cache/nonuniform/demand_n32_zipf1.0_seed42.txt

run() {  # run TAG NV CHUNK DEMAND EXTRA...
  local TAG=$1 NV=$2 CHUNK=$3 DEM=$4; shift 4
  local R
  R=$($BIN $COMMON -flex_nv_gbps $NV -chunk_size $CHUNK ${DEM:+-demand_file $DEM} "$@" 2>&1 \
      | grep -oE "validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+" \
      | tr '\n' ' ')
  echo "TAG $TAG nv $NV $R" >> $LOG
}

variant_flags() {
  case $1 in
    hp)       echo "-mosaic_hop_progress";;
    skewhp)   echo "-mosaic_skew_epoch 2 -mosaic_hop_progress";;
    rfhp)     echo "-mosaic_donate_rewaterfill -mosaic_hop_progress";;
    skewrfhp) echo "-mosaic_skew_epoch 2 -mosaic_donate_rewaterfill -mosaic_hop_progress";;
  esac
}

for NV in 90 180 360 2400; do
  for V in hp skewhp rfhp skewrfhp; do
    run "zipf1.0_seed42_$V" $NV 33554432 $D10 $(variant_flags $V)
  done
done
# robustness for the two E5 combos
for SD in 43 44; do
  for NV in 90 360; do
    for V in skewhp skewrfhp; do
      run "zipf1.0_seed${SD}_$V" $NV 33554432 cache/nonuniform/demand_n32_zipf1.0_seed${SD}.txt $(variant_flags $V)
    done
  done
done
for AL in 1.5 2.0; do
  for NV in 90 360; do
    for V in skewhp skewrfhp; do
      run "zipf${AL}_seed42_$V" $NV 67108864 cache/nonuniform/demand_n32_zipf${AL}_seed42.txt $(variant_flags $V)
    done
  done
done
for V in skewhp skewrfhp; do
  run "khot2_seed42_$V" 360 16777216 cache/nonuniform/demand_n32_khot2_seed42.txt $(variant_flags $V)
  run "persrc1.0_seed42_$V" 360 33554432 cache/nonuniform/demand_n32_persrc1.0_seed42.txt $(variant_flags $V)
  run "zipf1.0_seed42_degr26x4_$V" 2400 33554432 $D10 -flex_degrade_gpu 26 -flex_degrade_factor 4 $(variant_flags $V)
done
# uniform no-harm with full stack
for NV in 360 2400; do
  run "uniform_skewrfhp" $NV 2097152 "" -message_size 2097152 $(variant_flags skewrfhp)
done
# regression addendum: degrade C4 flags-off (archive 19组: 9253.09)
run "reg_degr26x4_c4" 2400 33554432 $D10 -flex_degrade_gpu 26 -flex_degrade_factor 4 -mosaic_corridors 4
echo DONE >> $LOG
