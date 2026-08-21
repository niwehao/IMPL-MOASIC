#!/bin/bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
for NV in 90 180 360; do
  ./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail -flex_nv_gbps $NV \
    -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule -mosaic_corridors 6 \
    -mosaic_trace -workload A2A -all_to_all -store_forward -quiet -demand_file $D \
    > cache/nonuniform/x26_trace_nv${NV}.log 2>&1
done
echo DONE >> cache/nonuniform/x26_trace_run.sh.log
