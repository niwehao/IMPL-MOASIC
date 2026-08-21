#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x29_explore.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -chunk_size 33554432 -demand_file $DZ -scheduler mosaic_schedule -mosaic_corridors 6"
run() { local tag="$1"; shift
  echo "$tag $($BIN $COMMON "$@" 2>&1 | grep -E '^validation' | sed 's/ flows.*queue_drops/ drops/;s/ split.*$//')" >> $LOG; }
: > $LOG
P="-topology flex_cube"
# init/prior variants on mainline
run e_niccap      $P -mosaic_cube_nic_cap
run e_lanerot     $P -mosaic_cube_lane_rot
run e_niccap_rot  $P -mosaic_cube_nic_cap -mosaic_cube_lane_rot
run e_uninit_rot  $P -mosaic_uniform_init -mosaic_cube_lane_rot
# wave count
run e_w8          $P -mosaic_waves 8
run e_w12         $P -mosaic_waves 12
run e_w8_niccap   $P -mosaic_waves 8 -mosaic_cube_nic_cap
# E5 censoring fix
run e_hopprog     $P -mosaic_hop_progress
run e_hopprog_niccap $P -mosaic_hop_progress -mosaic_cube_nic_cap
# combos with R mechanisms
run e_r_niccap    $P -mosaic_cube_nic_cap -mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill
# C4 anomaly retest with nic_cap
run e_c4_niccap   -topology flex_cube -mosaic_cube_nic_cap -mosaic_corridors 4
# transfer check: best variants on other regimes
for V in "-mosaic_cube_nic_cap" "-mosaic_cube_nic_cap -mosaic_cube_lane_rot"; do
  T=$(echo $V | tr -cd 'a-z_' | tail -c 12)
  run e_rot8_$T  -topology flex_cube -flex_cube_mix_rot 8 $V
  run e_100_$T   -topology flex_cube -flex_cube_speeds 100x16 $V
  run e_400_$T   -topology flex_cube -flex_cube_speeds 400x16 $V
done
run e_100_r_niccap -topology flex_cube -flex_cube_speeds 100x16 -mosaic_cube_nic_cap -mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill
echo DONE >> $LOG
