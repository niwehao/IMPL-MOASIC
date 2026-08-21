#!/bin/bash
# x28 phase-4: MOSAIC-R (coldE1+skew2+rewf) robustness suite
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x28_phase4.log
: > $LOG
BIN=./sim/tests/htsim_smart_regular
BASEOPT="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6"
run() { local TAG=$1; shift
  R=$($BIN $BASEOPT "$@" 2>&1 \
      | grep -oE "validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+" \
      | tr '\n' ' ')
  echo "TAG $TAG $R" >> $LOG
}
RFLAGS="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt

for SD in 43 44; do
  DS=cache/nonuniform/demand_n32_zipf1.0_seed${SD}.txt
  run "seed${SD}_90_R"  -flex_nv_gbps 90  -chunk_size 33554432 -demand_file $DS $RFLAGS
  run "seed${SD}_360_R" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file $DS $RFLAGS
done
for A in 1.5 2.0; do
  DA=cache/nonuniform/demand_n32_zipf${A}_seed42.txt
  run "zipf${A}_90_R"  -flex_nv_gbps 90  -chunk_size 67108864 -demand_file $DA $RFLAGS
  run "zipf${A}_360_R" -flex_nv_gbps 360 -chunk_size 67108864 -demand_file $DA $RFLAGS
done
run "khot2_360_R" -flex_nv_gbps 360 -chunk_size 16777216 -demand_file cache/nonuniform/demand_n32_khot2_seed42.txt $RFLAGS
run "degr26x4_2400_R" -flex_nv_gbps 2400 -flex_degrade_gpu 26 -flex_degrade_factor 4 -chunk_size 33554432 -demand_file $D $RFLAGS
run "taper90_360_R" -flex_nv_gbps 360 -flex_taper_gbps 90 -chunk_size 33554432 -demand_file $D $RFLAGS
run "persrc360_R" -flex_nv_gbps 360 -chunk_size 33554432 -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt $RFLAGS
run "uni360_R" -flex_nv_gbps 360 -chunk_size 2097152 -message_size 2097152 $RFLAGS
run "uni2400_R" -flex_nv_gbps 2400 -chunk_size 2097152 -message_size 2097152 $RFLAGS
# mainline mid nv for the final table (auto gate)
for NV in 420 480 600 720; do
  run "nv${NV}_R" -flex_nv_gbps $NV -chunk_size 33554432 -demand_file $D $RFLAGS
done
echo DONE >> $LOG
