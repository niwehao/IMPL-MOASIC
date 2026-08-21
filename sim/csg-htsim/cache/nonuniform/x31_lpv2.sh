#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x31_lpv2.log
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
# --- FAST-win points, zipf ---
run k_400_k6_w12    -topology flex_cube -flex_cube_speeds 400x16 $NZ -mosaic_waves 12 -mosaic_lp_spread $LPD/jd_400_k6.txt
run k_250_k6        -topology flex_cube -flex_cube_speeds 250x16 $NZ -mosaic_lp_spread $LPD/jd_250_k6.txt
run k_250_k6_w12    -topology flex_cube -flex_cube_speeds 250x16 $NZ -mosaic_waves 12 -mosaic_lp_spread $LPD/jd_250_k6.txt
run k_full50_k6     -topology flex_cube -flex_cube_full_gbps 50 $NZ -mosaic_lp_spread $LPD/jd_full50_k6.txt
run k_full300_k6_w12 -topology flex_cube -flex_cube_full_gbps 300 $NZ -mosaic_waves 12 -mosaic_lp_spread $LPD/jd_full300_k6.txt
# --- FAST-win points, uniform demand ---
run k_uni400_k6     -topology flex_cube -flex_cube_speeds 400x16 $UNI -mosaic_lp_spread $LPD/jd_400_k6.txt
run k_unifull300_k6 -topology flex_cube -flex_cube_full_gbps 300 $UNI -mosaic_lp_spread $LPD/jd_full300_k6.txt
run k_unimix_k6     -topology flex_cube $UNI -mosaic_lp_spread $LPD/jd_mix_rot0_k6.txt
# --- heterogeneous mainline + rot (stacks) ---
run k_mix_k6        -topology flex_cube $NZ -mosaic_lp_spread $LPD/jd_mix_rot0_k6.txt
run k_mix_k6_w12    -topology flex_cube $NZ -mosaic_waves 12 -mosaic_lp_spread $LPD/jd_mix_rot0_k6.txt
run k_mix_full_w16  -topology flex_cube $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jd_mix_rot0_k6.txt
run k_rot4_stack    -topology flex_cube -flex_cube_mix_rot 4 $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot4.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jd_mix_rot4_k6.txt
run k_rot8_stack    -topology flex_cube -flex_cube_mix_rot 8 $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot8.txt -mosaic_lp_spread $LPD/jd_mix_rot8_k6.txt
run k_rot12_stack   -topology flex_cube -flex_cube_mix_rot 12 $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot12.txt -mosaic_lp_spread $LPD/jd_mix_rot12_k6.txt
# --- NV-scarce profiles ---
run k_100_w16_stack -topology flex_cube -flex_cube_speeds 100x16 $NZ -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_100.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jd_100_k6.txt
run k_100_r_stack   -topology flex_cube -flex_cube_speeds 100x16 $NZ $R_FLAGS -mosaic_lp_intra $LPD/cube_100.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jd_100_k6.txt
run k_50_r_stack    -topology flex_cube -flex_cube_speeds 50x16 $NZ $R_FLAGS -mosaic_lp_intra $LPD/cube_50.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jd_50_k6.txt
# --- flex_rail (NVSwitch): unified LP on star graph ---
run k_rail90_jd     -topology flex_rail -flex_nv_gbps 90 $NZ -mosaic_lp_spread $LPD/jd_rail90_k6.txt
run k_rail360_jd    -topology flex_rail -flex_nv_gbps 360 $NZ -mosaic_lp_spread $LPD/jd_rail360_k6.txt
run k_rail2400_jd   -topology flex_rail -flex_nv_gbps 2400 $NZ -mosaic_lp_spread $LPD/jd_rail2400_k6.txt
run k_rail360_jd_R  -topology flex_rail -flex_nv_gbps 360 $NZ $R_FLAGS -mosaic_lp_spread $LPD/jd_rail360_k6.txt
run k_rail90_jd_R   -topology flex_rail -flex_nv_gbps 90 $NZ $R_FLAGS -mosaic_lp_spread $LPD/jd_rail90_k6.txt
# --- robustness ---
run k_seed43_stack  -topology flex_cube -chunk_size 33554432 -demand_file cache/nonuniform/demand_n32_zipf1.0_seed43.txt -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jd_mix_rot0_k6.txt
run k_zipf1.5_stack -topology flex_cube -chunk_size 67108864 -demand_file cache/nonuniform/demand_n32_zipf1.5_seed42.txt -mosaic_waves 16 -mosaic_lp_intra $LPD/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread $LPD/jd_mix_rot0_k6.txt
echo DONE >> $LOG
