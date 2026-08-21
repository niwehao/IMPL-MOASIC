#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x29_explore2.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
NONUNI="-chunk_size 33554432 -demand_file $DZ"
run() { local tag="$1"; shift
  echo "$tag $($BIN $COMMON "$@" 2>&1 | grep -E '^validation' | sed 's/ flows.*queue_drops/ drops/;s/ split.*$//')" >> $LOG; }
: > $LOG
# W12 transfer + combos
run f_w12_rot8    -topology flex_cube -flex_cube_mix_rot 8 $NONUNI -mosaic_waves 12
run f_w12_100     -topology flex_cube -flex_cube_speeds 100x16 $NONUNI -mosaic_waves 12
run f_w12_400     -topology flex_cube -flex_cube_speeds 400x16 $NONUNI -mosaic_waves 12
run f_w12_50      -topology flex_cube -flex_cube_speeds 50x16 $NONUNI -mosaic_waves 12
run f_w12_rot4    -topology flex_cube -flex_cube_mix_rot 4 $NONUNI -mosaic_waves 12
run f_w12_rot12   -topology flex_cube -flex_cube_mix_rot 12 $NONUNI -mosaic_waves 12
run f_w12_lanerot -topology flex_cube $NONUNI -mosaic_waves 12 -mosaic_cube_lane_rot
run f_w12_lanerot_100 -topology flex_cube -flex_cube_speeds 100x16 $NONUNI -mosaic_waves 12 -mosaic_cube_lane_rot
run f_w12_lanerot_400 -topology flex_cube -flex_cube_speeds 400x16 $NONUNI -mosaic_waves 12 -mosaic_cube_lane_rot
run f_w12_lanerot_rot8 -topology flex_cube -flex_cube_mix_rot 8 $NONUNI -mosaic_waves 12 -mosaic_cube_lane_rot
run f_w12_r_100   -topology flex_cube -flex_cube_speeds 100x16 $NONUNI -mosaic_waves 12 -mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill
run f_w12_uni     -topology flex_cube -chunk_size 2097152 -message_size 2097152 -mosaic_waves 12
# W sweep granularity on mainline
run f_w9          -topology flex_cube $NONUNI -mosaic_waves 9
run f_w16         -topology flex_cube $NONUNI -mosaic_waves 16
run f_w24         -topology flex_cube $NONUNI -mosaic_waves 24
# rail sanity: does W12 also help flex_rail@360? (context for "cube-specific or not")
run f_w12_rail360 -topology flex_rail -flex_nv_gbps 360 $NONUNI -mosaic_waves 12
echo DONE >> $LOG
