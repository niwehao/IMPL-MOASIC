#!/bin/bash
# x32_p1rail: 问题1 rail 侧诊断 -- taper 全扫 x 变体、solo 对照、
# demand 稳健性、per-dst trace。冻结二进制。
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
B=cache/nonuniform/x32bin/htsim_f1f5b2a
LOG=cache/nonuniform/x32_p1rail.log
{ echo "# binary $(md5sum $B)"; } > $LOG
Z=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
C="-nodes 32 -topology flex_rail -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule -mosaic_corridors 6"
R="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
run(){ local TAG=$1; shift; local OUT; OUT=$($B $C "$@" 2>&1)
  echo "$TAG $(echo "$OUT" | grep -oE 'validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+') $(echo "$OUT" | grep -oE 'p99_us [0-9.]+') $(echo "$OUT" | grep -oE 'hop_bytes [0-9]+')" >> $LOG; }
# (a) taper x variants, zipf1.0/seed42
for T in 45 90 180 360 720 1440; do
  run "zipf t$T base"   -flex_taper_gbps $T -demand_file $Z
  run "zipf t$T static" -flex_taper_gbps $T -demand_file $Z -mosaic_static
  run "zipf t$T R"      -flex_taper_gbps $T -demand_file $Z $R
done
# (b) solo hot-dst control (无争用)
for T in 45 90 180 360 720 1440; do
  run "solo t$T" -flex_taper_gbps $T -demand_file cache/nonuniform/demand_n32_solo_dst26.txt
done
# (c) demand robustness
for DM in seed43:demand_n32_zipf1.0_seed43.txt zipf15:demand_n32_zipf1.5_seed42.txt \
          persrc:demand_n32_persrc1.0_seed42.txt uniform:demand_n32_uniform2mb.txt; do
  N=${DM%%:*}; F=cache/nonuniform/${DM#*:}
  for T in 45 90 180 360 720 1440; do
    run "$N t$T" -flex_taper_gbps $T -demand_file $F
  done
done
# (d) per-dst trace 归因(完整日志落盘)
for T in 90 180 360; do
  $B $C -flex_taper_gbps $T -demand_file $Z -mosaic_trace > cache/nonuniform/x32_trace_t$T.log 2>&1
done
echo DONE >> $LOG
