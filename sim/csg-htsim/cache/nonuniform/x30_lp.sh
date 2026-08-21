#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x30_lp.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
LPD=cache/mosaic_lp
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
NONUNI="-chunk_size 33554432 -demand_file $DZ"
R_FLAGS="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
run() { local tag="$1"; shift
  echo "$tag $($BIN $COMMON "$@" 2>&1 | grep -E '^validation' | sed 's/ flows.*queue_drops/ drops/;s/ split.*$//')" >> $LOG; }
: > $LOG
# mainline + rot sweep, SOTA W16, lp vs lp_local
for ROT in 0 4 8 12; do
  T="-topology flex_cube -flex_cube_mix_rot $ROT"
  LP="$LPD/cube_mix_rot${ROT}.txt"
  run h_rot${ROT}_lp       $T $NONUNI -mosaic_waves 16 -mosaic_lp_intra $LP
  run h_rot${ROT}_lploc    $T $NONUNI -mosaic_waves 16 -mosaic_lp_intra $LP -mosaic_lp_local
done
# homogeneous profiles (per-profile SOTA W)
run h_100_w6_lploc  -topology flex_cube -flex_cube_speeds 100x16 $NONUNI -mosaic_lp_intra $LPD/cube_100.txt -mosaic_lp_local
run h_100_r_lploc   -topology flex_cube -flex_cube_speeds 100x16 $NONUNI $R_FLAGS -mosaic_lp_intra $LPD/cube_100.txt -mosaic_lp_local
run h_100_w16_lploc -topology flex_cube -flex_cube_speeds 100x16 $NONUNI -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_100.txt -mosaic_lp_local
run h_400_w6_lploc  -topology flex_cube -flex_cube_speeds 400x16 $NONUNI -mosaic_lp_intra $LPD/cube_400.txt -mosaic_lp_local
run h_400_w12_lploc -topology flex_cube -flex_cube_speeds 400x16 $NONUNI -mosaic_waves 12 -mosaic_lp_intra $LPD/cube_400.txt -mosaic_lp_local
run h_250_w6_lploc  -topology flex_cube -flex_cube_speeds 250x16 $NONUNI -mosaic_lp_intra $LPD/cube_250.txt -mosaic_lp_local
run h_50_w6_lploc   -topology flex_cube -flex_cube_speeds 50x16 $NONUNI -mosaic_lp_intra $LPD/cube_50.txt -mosaic_lp_local
run h_50_r_lploc    -topology flex_cube -flex_cube_speeds 50x16 $NONUNI $R_FLAGS -mosaic_lp_intra $LPD/cube_50.txt -mosaic_lp_local
# degenerate sanity (full crossbar: LP paths are all direct -> expect == base)
run h_full300_lploc -topology flex_cube -flex_cube_full_gbps 300 $NONUNI -mosaic_lp_intra $LPD/cube_full300.txt -mosaic_lp_local
# uniform demand
run h_uni_lploc     -topology flex_cube -chunk_size 2097152 -message_size 2097152 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local
# robustness
run h_seed43_lploc  -topology flex_cube -chunk_size 33554432 -demand_file cache/nonuniform/demand_n32_zipf1.0_seed43.txt -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local
run h_zipf1.5_lploc -topology flex_cube -chunk_size 67108864 -demand_file cache/nonuniform/demand_n32_zipf1.5_seed42.txt -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local
# W interaction on mainline: does LP reduce the need for many waves?
run h_w6_lploc      -topology flex_cube $NONUNI -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local
run h_w12_lploc     -topology flex_cube $NONUNI -mosaic_waves 12 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local
run h_w24_lploc     -topology flex_cube $NONUNI -mosaic_waves 24 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local
echo DONE >> $LOG
