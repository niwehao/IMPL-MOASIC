#!/bin/bash
# x26 regression (handoff §6, flags-off must match archive) + W cross-validation
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x26_regression.log
: > $LOG
BIN=./sim/tests/htsim_smart_regular
D10=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
NU="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet"

run() { local TAG=$1; shift
  R=$($BIN "$@" 2>&1 \
      | grep -oE "validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+" \
      | tr '\n' ' ')
  echo "TAG $TAG $R" >> $LOG
}

# §6.2 main-line 8 nv points x C6 (90/180/360/2400 covered by matrix base)
for NV in 420 480 600 720; do
  run "reg_c6_nv$NV" $NU -topology flex_rail -flex_nv_gbps $NV -chunk_size 33554432 \
    -scheduler mosaic_schedule -mosaic_corridors 6 -demand_file $D10
done
# §6.4 scene checks (flags off, expect archived numbers)
run "reg_taper90_c4" $NU -topology flex_rail -flex_nv_gbps 360 -flex_taper_gbps 90 \
  -chunk_size 33554432 -scheduler mosaic_schedule -demand_file $D10
run "reg_pod360_l720_c5" $NU -topology flex_pod -flex_spines 4 -flex_nv_gbps 360 \
  -flex_taper_gbps 720 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 5 -demand_file $D10
run "reg_r2p_b4_c6" $NU -topology flex_rail2p -flex_nv_gbps 2400 -flex_planeb_factor 4 \
  -chunk_size 33554432 -scheduler mosaic_schedule -mosaic_corridors 6 -demand_file $D10
run "reg_degr26x4_c6" $NU -topology flex_rail -flex_nv_gbps 2400 -flex_degrade_gpu 26 \
  -flex_degrade_factor 4 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -demand_file $D10
# W cross-validation @ nv90 C6: base vs skew2+rewf, W in {2,4,6,8,12}
for W in 2 4 6 8 12; do
  run "wsweep_base_W$W" $NU -topology flex_rail -flex_nv_gbps 90 -chunk_size 33554432 \
    -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_waves $W -demand_file $D10
  run "wsweep_skewrf_W$W" $NU -topology flex_rail -flex_nv_gbps 90 -chunk_size 33554432 \
    -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_waves $W \
    -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill -demand_file $D10
done
echo DONE >> $LOG
