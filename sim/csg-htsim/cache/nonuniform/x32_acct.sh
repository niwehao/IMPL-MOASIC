#!/bin/bash
# x32_acct: 挂账重做 -- 23 组 rail2p 全网格重跑(当前代码)+ 24 组 C5 复核。
# 一律用冻结二进制,保证批内一致。
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
B=cache/nonuniform/x32bin/htsim_f1f5b2a
LOG=cache/nonuniform/x32_acct.log
{ echo "# binary $(md5sum $B)"; echo "# git $(git log --oneline -1 | cut -d' ' -f1)"; } > $LOG
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
C="-nodes 32 -topology flex_rail2p -packet_size 1024 -chunk_size 33554432 -workload A2A -all_to_all -store_forward -quiet -demand_file $D"
run(){ local TAG=$1; shift
  echo "$TAG $($B $C "$@" 2>&1 | grep -oE 'validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+' | tr '\n' ' ')" >> $LOG; }
for NV in 2400 360 90; do for PB in 1 2 4; do
  run "nv$NV pb$PB ecmp" -flex_nv_gbps $NV -flex_planeb_factor $PB -scheduler static_ecmp
  run "nv$NV pb$PB fast" -flex_nv_gbps $NV -flex_planeb_factor $PB -scheduler fast_schedule -fast_alpha_beta_guard
  run "nv$NV pb$PB c4"   -flex_nv_gbps $NV -flex_planeb_factor $PB -scheduler mosaic_schedule -mosaic_corridors 4
  run "nv$NV pb$PB c6"   -flex_nv_gbps $NV -flex_planeb_factor $PB -scheduler mosaic_schedule -mosaic_corridors 6
done; done
# 24 组 X4(C5)复核:自身归档是否可复现
for NV in 2400 360 90; do for PB in 1 2 4; do
  run "x4 nv$NV pb$PB c5" -flex_nv_gbps $NV -flex_planeb_factor $PB -scheduler mosaic_schedule -mosaic_corridors 5
  if [ $PB -gt 1 ]; then
    run "x4 nv$NV pb$PB c5known" -flex_nv_gbps $NV -flex_planeb_factor $PB -scheduler mosaic_schedule -mosaic_corridors 5 -mosaic_known_planeb $PB
  fi
done; done
echo DONE >> $LOG
