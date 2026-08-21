#!/bin/bash
# x32_p1pod: 问题1 pod 侧 -- spine 数 x corridor 数解耦网格。冻结二进制。
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
B=cache/nonuniform/x32bin/htsim_f1f5b2a
LOG=cache/nonuniform/x32_p1pod.log
{ echo "# binary $(md5sum $B)"; } > $LOG
Z=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
C="-nodes 32 -topology flex_pod -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -demand_file $Z"
run(){ local TAG=$1; shift
  echo "$TAG $($B $C "$@" 2>&1 | grep -oE 'validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+')" >> $LOG; }
# (a) 总上行固定 720(带宽不变,只改并联度)
for S in 1 2 3 4 5 6; do for CC in 2 3 4 5 6; do
  run "fix S$S C$CC imp" -flex_taper_gbps 720 -flex_spines $S -mosaic_corridors $CC
  run "fix S$S C$CC exp" -flex_taper_gbps 720 -flex_spines $S -mosaic_corridors $CC -mosaic_explicit_spine
done; done
# (b) 每条 spine 固定 180(带宽随 S 增长)
for S in 1 2 3 4 5 6; do for CC in 4 5 6; do
  run "grow S$S C$CC imp" -flex_taper_gbps $((180*S)) -flex_spines $S -mosaic_corridors $CC
done; done
echo DONE >> $LOG
