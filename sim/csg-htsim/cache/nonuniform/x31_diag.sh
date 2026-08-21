#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x31_diag.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet"
NZ="-chunk_size 33554432 -demand_file $DZ"
UNI="-chunk_size 2097152 -message_size 2097152"
M="-scheduler mosaic_schedule -mosaic_corridors 6"
run() { local tag="$1"; shift
  echo "$tag $($BIN $COMMON "$@" 2>&1 | grep -E '^validation' | sed 's/ flows.*queue_drops/ drops/;s/ split.*$//')" >> $LOG; }
: > $LOG
# D2/D4: uniform demand on homogeneous point-to-point -- skew-specific or structural?
run d_400_uni_fast   -topology flex_cube -flex_cube_speeds 400x16 $UNI -scheduler fast_schedule -fast_alpha_beta_guard
run d_400_uni_mosaic -topology flex_cube -flex_cube_speeds 400x16 $UNI $M
run d_full300_uni_fast   -topology flex_cube -flex_cube_full_gbps 300 $UNI -scheduler fast_schedule -fast_alpha_beta_guard
run d_full300_uni_mosaic -topology flex_cube -flex_cube_full_gbps 300 $UNI $M
# D5: corridor ablation on 400x16 zipf (lane concentration suspicion)
for C in 2 3 4; do run d_400_c$C -topology flex_cube -flex_cube_speeds 400x16 $NZ -scheduler mosaic_schedule -mosaic_corridors $C; done
# D1: feedback on/off at full300
run d_full300_static -topology flex_cube -flex_cube_full_gbps 300 $NZ $M -mosaic_static
# best-known stacks on 400x16 / full300
run d_400_rot_w12_lp -topology flex_cube -flex_cube_speeds 400x16 $NZ $M -mosaic_waves 12 -mosaic_cube_lane_rot -mosaic_lp_intra cache/mosaic_lp/cube_400.txt -mosaic_lp_local
run d_full300_lanerot -topology flex_cube -flex_cube_full_gbps 300 $NZ $M -mosaic_cube_lane_rot
run d_full300_rot_w12 -topology flex_cube -flex_cube_full_gbps 300 $NZ $M -mosaic_waves 12 -mosaic_cube_lane_rot
echo DONE >> $LOG
