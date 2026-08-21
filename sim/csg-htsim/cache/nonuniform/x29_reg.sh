#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x29_reg.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
: > $LOG
for NV in 90 180 360 420 480 600 720 2400; do
  US=$($BIN -nodes 32 -topology flex_rail -flex_nv_gbps $NV -packet_size 1024 -chunk_size 33554432 \
    -scheduler mosaic_schedule -mosaic_corridors 6 -workload A2A -all_to_all -store_forward -quiet \
    -demand_file $DZ 2>&1 | grep -oE "completion_us [0-9.]+")
  echo "nv$NV $US" >> $LOG
done
echo DONE >> $LOG
