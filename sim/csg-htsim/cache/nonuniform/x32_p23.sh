#!/bin/bash
# x32_p23: 问题2/3 中不需要新旋钮的诊断 -- 19 组效应在各配置下复核、
# 单卡双向降速矩阵、扩 C 证伪。冻结二进制(committed 版只支持双向降速)。
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
B=cache/nonuniform/x32bin/htsim_f1f5b2a
LOG=cache/nonuniform/x32_p23.log
{ echo "# binary $(md5sum $B)"; } > $LOG
Z=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
C="-nodes 32 -topology flex_rail -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 -workload A2A -all_to_all -store_forward -quiet -demand_file $Z"
R="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
run(){ local TAG=$1; shift
  echo "$TAG $($B $C "$@" 2>&1 | grep -oE 'validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+')" >> $LOG; }
# (a) 无降速参照
run "ref c4"   -scheduler mosaic_schedule -mosaic_corridors 4
run "ref c6"   -scheduler mosaic_schedule -mosaic_corridors 6
run "ref c6R"  -scheduler mosaic_schedule -mosaic_corridors 6 $R
run "ref fast" -scheduler fast_schedule -fast_alpha_beta_guard
run "ref ecmp" -scheduler static_ecmp
# (b) 降速矩阵(双向): gpu {7 冷源, 2 中继, 24 热点同机冷dst, 26 热点} x {2,4}
for G in 7 2 24 26; do for F in 2 4; do
  DG="-flex_degrade_gpu $G -flex_degrade_factor $F"
  run "g$G f$F c4"   $DG -scheduler mosaic_schedule -mosaic_corridors 4
  run "g$G f$F c6"   $DG -scheduler mosaic_schedule -mosaic_corridors 6
  run "g$G f$F c6R"  $DG -scheduler mosaic_schedule -mosaic_corridors 6 $R
  run "g$G f$F fast" $DG -scheduler fast_schedule -fast_alpha_beta_guard
done; done
# (c) 问题3: 扩 C 能否救 g2x4(中继降速)
for CC in 2 3 4 5 6; do
  run "g2f4 C$CC" -flex_degrade_gpu 2 -flex_degrade_factor 4 -scheduler mosaic_schedule -mosaic_corridors $CC
done
for RS in 2 3; do
  run "g2f4 C6 ramp$RS" -flex_degrade_gpu 2 -flex_degrade_factor 4 -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_ramp_start $RS
done
echo DONE >> $LOG
