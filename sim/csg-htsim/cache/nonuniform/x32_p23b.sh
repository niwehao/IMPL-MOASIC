#!/bin/bash
# x32_p23b: 问题2/3 需要仪器旋钮的诊断 -- 方向分解、单中继 vs 全类中继、
# oracle 冷入口限速。用仪器冻结二进制 htsim_instr。
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
B=cache/nonuniform/x32bin/htsim_instr
LOG=cache/nonuniform/x32_p23b.log
{ echo "# binary $(md5sum $B)"; } > $LOG
Z=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
C="-nodes 32 -topology flex_rail -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 -workload A2A -all_to_all -store_forward -quiet -demand_file $Z"
R="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
run(){ local TAG=$1; shift
  echo "$TAG $($B $C "$@" 2>&1 | grep -oE 'validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+')" >> $LOG; }
# (a) 方向分解(dir=0 行在 x32_p23)
for G in 7 2 24 26; do for F in 2 4; do for DIR in 1 2; do
  DG="-flex_degrade_gpu $G -flex_degrade_factor $F -flex_degrade_dir $DIR"
  run "g$G f$F d$DIR c4"   $DG -scheduler mosaic_schedule -mosaic_corridors 4
  run "g$G f$F d$DIR c6"   $DG -scheduler mosaic_schedule -mosaic_corridors 6
  run "g$G f$F d$DIR c6R"  $DG -scheduler mosaic_schedule -mosaic_corridors 6 $R
  run "g$G f$F d$DIR fast" $DG -scheduler fast_schedule -fast_alpha_beta_guard
done; done; done
# (b) 问题3: 单中继 vs 双中继 vs 全类中继(出口降 4x)
for SET in 2 2,10 2,10,18; do
  T=$(echo $SET | tr ',' '_')
  DG="-flex_degrade_gpus $SET -flex_degrade_factor 4 -flex_degrade_dir 1"
  run "relay$T c6"   $DG -scheduler mosaic_schedule -mosaic_corridors 6
  run "relay$T c6R"  $DG -scheduler mosaic_schedule -mosaic_corridors 6 $R
  run "relay$T fast" $DG -scheduler fast_schedule -fast_alpha_beta_guard
done
# (c) oracle: 把冷 dst 入口整体限速(除 26;以及除 26+同轨2/10/18)
ALL=$(python3 -c "print(','.join(str(i) for i in range(32) if i!=26))")
NR2=$(python3 -c "print(','.join(str(i) for i in range(32) if i not in (26,2,10,18)))")
for F in 1.25 1.5 2 4; do
  for M in "allbut26:$ALL" "allbutrail2:$NR2"; do
    N=${M%%:*}; L=${M#*:}
    run "oracle_$N f$F c6"  -flex_degrade_gpus $L -flex_degrade_factor $F -flex_degrade_dir 2 -scheduler mosaic_schedule -mosaic_corridors 6
    run "oracle_$N f$F c6R" -flex_degrade_gpus $L -flex_degrade_factor $F -flex_degrade_dir 2 -scheduler mosaic_schedule -mosaic_corridors 6 $R
  done
done
echo DONE >> $LOG
