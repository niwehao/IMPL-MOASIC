#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x29_explore3.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
NONUNI="-chunk_size 33554432 -demand_file $DZ"
run() { local tag="$1"; shift
  echo "$tag $($BIN $COMMON "$@" 2>&1 | grep -E '^validation' | sed 's/ flows.*queue_drops/ drops/;s/ split.*$//')" >> $LOG; }
: > $LOG
run g_w16_rot8  -topology flex_cube -flex_cube_mix_rot 8 $NONUNI -mosaic_waves 16
run g_w16_rot4  -topology flex_cube -flex_cube_mix_rot 4 $NONUNI -mosaic_waves 16
run g_w16_rot12 -topology flex_cube -flex_cube_mix_rot 12 $NONUNI -mosaic_waves 16
run g_w16_100   -topology flex_cube -flex_cube_speeds 100x16 $NONUNI -mosaic_waves 16
run g_w16_400   -topology flex_cube -flex_cube_speeds 400x16 $NONUNI -mosaic_waves 16
run g_w16_50    -topology flex_cube -flex_cube_speeds 50x16 $NONUNI -mosaic_waves 16
run g_w16_250   -topology flex_cube -flex_cube_speeds 250x16 $NONUNI -mosaic_waves 16
run g_w16_uni   -topology flex_cube -chunk_size 2097152 -message_size 2097152 -mosaic_waves 16
run g_w16_r_100 -topology flex_cube -flex_cube_speeds 100x16 $NONUNI -mosaic_waves 16 -mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill
run g_w12_250   -topology flex_cube -flex_cube_speeds 250x16 $NONUNI -mosaic_waves 12
# robustness of recommended W16 on seeds/alpha (mainline profile)
for D2 in demand_n32_zipf1.0_seed43.txt demand_n32_zipf1.5_seed42.txt; do
  if [ -f cache/nonuniform/$D2 ]; then
    run g_w16_${D2%.txt} -topology flex_cube -chunk_size 67108864 -demand_file cache/nonuniform/$D2 -mosaic_waves 16
    run g_base_${D2%.txt} -topology flex_cube -chunk_size 67108864 -demand_file cache/nonuniform/$D2
  fi
done
echo DONE >> $LOG
