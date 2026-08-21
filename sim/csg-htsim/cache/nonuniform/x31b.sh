#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x31b.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
LPD=cache/mosaic_lp
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
NZ="-chunk_size 33554432 -demand_file $DZ"
UNI="-chunk_size 2097152 -message_size 2097152"
R_FLAGS="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
run() { local tag="$1"; shift
  echo "$tag $($BIN $COMMON "$@" 2>&1 | grep -E '^validation' | sed 's/ flows.*queue_drops/ drops/;s/ split.*$//')" >> $LOG; }
: > $LOG
# homogeneous (FAST-win) points, zipf
run m_400_jh       -topology flex_cube -flex_cube_speeds 400x16 $NZ -mosaic_lp_spread $LPD/jh_400.txt
run m_400_jh_w12   -topology flex_cube -flex_cube_speeds 400x16 $NZ -mosaic_waves 12 -mosaic_lp_spread $LPD/jh_400.txt
run m_250_jh       -topology flex_cube -flex_cube_speeds 250x16 $NZ -mosaic_lp_spread $LPD/jh_250.txt
run m_250_jh_w12   -topology flex_cube -flex_cube_speeds 250x16 $NZ -mosaic_waves 12 -mosaic_lp_spread $LPD/jh_250.txt
run m_full300_jh   -topology flex_cube -flex_cube_full_gbps 300 $NZ -mosaic_lp_spread $LPD/jh_full300.txt
run m_full50_jh    -topology flex_cube -flex_cube_full_gbps 50 $NZ -mosaic_lp_spread $LPD/jh_full50.txt
# uniform demand
run m_uni400_jh    -topology flex_cube -flex_cube_speeds 400x16 $UNI -mosaic_lp_spread $LPD/jh_400.txt
run m_unifull300_jh -topology flex_cube -flex_cube_full_gbps 300 $UNI -mosaic_lp_spread $LPD/jh_full300.txt
run m_unimix_jh    -topology flex_cube $UNI -mosaic_lp_spread $LPD/jh_mix_rot0.txt
# heterogeneous stacks (must beat/match 30 组 SOTA)
run m_mix_stack_w16 -topology flex_cube $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jh_mix_rot0.txt
run m_mix_stack_w12 -topology flex_cube $NZ -mosaic_waves 12 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jh_mix_rot0.txt
run m_rot4_stack   -topology flex_cube -flex_cube_mix_rot 4 $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot4.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jh_mix_rot4.txt
run m_rot8_stack   -topology flex_cube -flex_cube_mix_rot 8 $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot8.txt -mosaic_lp_spread $LPD/jh_mix_rot8.txt
run m_rot12_stack  -topology flex_cube -flex_cube_mix_rot 12 $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot12.txt -mosaic_lp_spread $LPD/jh_mix_rot12.txt
run m_100_r_stack  -topology flex_cube -flex_cube_speeds 100x16 $NZ $R_FLAGS -mosaic_lp_intra $LPD/cube_100.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jh_100.txt
run m_50_r_stack   -topology flex_cube -flex_cube_speeds 50x16 $NZ $R_FLAGS -mosaic_lp_intra $LPD/cube_50.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jh_50.txt
run m_seed43_stack -topology flex_cube -chunk_size 33554432 -demand_file cache/nonuniform/demand_n32_zipf1.0_seed43.txt -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jh_mix_rot0.txt
run m_zipf1.5_stack -topology flex_cube -chunk_size 67108864 -demand_file cache/nonuniform/demand_n32_zipf1.5_seed42.txt -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jh_mix_rot0.txt
echo DONE >> $LOG
