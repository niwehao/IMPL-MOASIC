#!/bin/bash
# x32_mech: F6/F7 全面验证 -- rail 8 nv 点、门(uniform/persrc/taper)、
# 稳健性(seed/zipf/khot/drift/n64)、其他拓扑、降速矩阵下的 F7。
# 用机制冻结二进制 htsim_mech。
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
B=cache/nonuniform/x32bin/htsim_mech
LOG=cache/nonuniform/x32_mech.log
{ echo "# binary $(md5sum $B)"; } > $LOG
Z=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
BASE="-nodes 32 -packet_size 1024 -chunk_size 33554432 -workload A2A -all_to_all -store_forward -quiet"
R="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"
run(){ local TAG=$1; shift; local OUT; OUT=$($B $BASE "$@" 2>&1)
  echo "$TAG $(echo "$OUT" | grep -oE 'validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+') $(echo "$OUT" | grep -oE 'p99_us [0-9.]+') $(echo "$OUT" | grep -oE 'cold_pace (on|off)[a-z ]*') $(echo "$OUT" | grep -oE 'queue_drops [0-9]+')" >> $LOG; }
M="-topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6 -demand_file $Z"
# (a) rail 8 nv 点 x {R, pace1, pace2, R+pace1, hostλ}
for NV in 90 180 360 420 480 600 720 2400; do
  run "nv$NV R"      $M -flex_nv_gbps $NV $R
  run "nv$NV pace1"  $M -flex_nv_gbps $NV -mosaic_cold_pace 1
  run "nv$NV pace2"  $M -flex_nv_gbps $NV -mosaic_cold_pace 2
  run "nv$NV Rpace1" $M -flex_nv_gbps $NV $R -mosaic_cold_pace 1
  run "nv$NV hostl"  $M -flex_nv_gbps $NV -mosaic_host_lambda
done
# (b) 门:uniform/persrc(期望逐位=base)、taper(45/90 门3关)
for DM in uniform:demand_n32_uniform2mb.txt persrc:demand_n32_persrc1.0_seed42.txt; do
  N=${DM%%:*}; F=cache/nonuniform/${DM#*:}
  for NV in 90 360 2400; do
    run "$N nv$NV base"  -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6 -flex_nv_gbps $NV -demand_file $F
    run "$N nv$NV pace1" -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6 -flex_nv_gbps $NV -demand_file $F -mosaic_cold_pace 1
    run "$N nv$NV hostl" -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6 -flex_nv_gbps $NV -demand_file $F -mosaic_host_lambda
  done
done
for T in 45 90 180 360 720 1440; do
  run "taper$T pace1"  $M -flex_nv_gbps 360 -flex_taper_gbps $T -mosaic_cold_pace 1
  run "taper$T Rpace1" $M -flex_nv_gbps 360 -flex_taper_gbps $T $R -mosaic_cold_pace 1
done
# (c) 稳健性
for DM in seed43:demand_n32_zipf1.0_seed43.txt zipf15:demand_n32_zipf1.5_seed42.txt \
          khot2:demand_n32_khot2_seed42.txt khot4:demand_n32_khot4_seed42.txt khot8:demand_n32_khot8_seed42.txt; do
  N=${DM%%:*}; F=cache/nonuniform/${DM#*:}
  [ -f "$F" ] || { echo "$N missing" >> $LOG; continue; }
  run "$N base"  -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6 -flex_nv_gbps 360 -demand_file $F
  run "$N pace1" -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6 -flex_nv_gbps 360 -demand_file $F -mosaic_cold_pace 1
  run "$N R"     -topology flex_rail -scheduler mosaic_schedule -mosaic_corridors 6 -flex_nv_gbps 360 -demand_file $F $R
done
# drift: prior 用 seed43 矩阵、实跑 seed42
run "drift base"  $M -flex_nv_gbps 360 -mosaic_prior_demand_file cache/nonuniform/demand_n32_zipf1.0_seed43.txt
run "drift pace1" $M -flex_nv_gbps 360 -mosaic_prior_demand_file cache/nonuniform/demand_n32_zipf1.0_seed43.txt -mosaic_cold_pace 1
# n64
N64=cache/nonuniform/demand_n64_zipf1.0_seed42.txt
run "n64 base"  -nodes 64 -topology flex_rail -flex_hosts 8 -flex_gpus_per_host 8 -flex_nv_gbps 360 -scheduler mosaic_schedule -mosaic_corridors 6 -demand_file $N64
run "n64 pace1" -nodes 64 -topology flex_rail -flex_hosts 8 -flex_gpus_per_host 8 -flex_nv_gbps 360 -scheduler mosaic_schedule -mosaic_corridors 6 -demand_file $N64 -mosaic_cold_pace 1
# (d) 其他拓扑
run "pod up1440 pace1" -topology flex_pod -flex_nv_gbps 360 -flex_taper_gbps 1440 -flex_spines 4 -scheduler mosaic_schedule -mosaic_corridors 5 -demand_file $Z -mosaic_cold_pace 1
run "pod up720 pace1"  -topology flex_pod -flex_nv_gbps 360 -flex_taper_gbps 720 -flex_spines 4 -scheduler mosaic_schedule -mosaic_corridors 5 -demand_file $Z -mosaic_cold_pace 1
run "r2p pb4 pace1"    -topology flex_rail2p -flex_nv_gbps 2400 -flex_planeb_factor 4 -scheduler mosaic_schedule -mosaic_corridors 6 -demand_file $Z -mosaic_cold_pace 1
run "r2p sym pace1"    -topology flex_rail2p -flex_nv_gbps 360 -flex_planeb_factor 1 -scheduler mosaic_schedule -mosaic_corridors 6 -demand_file $Z -mosaic_cold_pace 1
CU="-topology flex_cube -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_waves 16 -mosaic_lp_intra cache/mosaic_lp/cube_mix_rot0.txt -mosaic_lp_local -mosaic_lp_spread cache/mosaic_lp/jh_mix_rot0.txt -demand_file $Z"
run "cube stack"       $CU
run "cube stack_pace1" $CU -mosaic_cold_pace 1
CU250="-topology flex_cube -flex_cube_speeds 250x16 -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_waves 12 -mosaic_lp_spread cache/mosaic_lp/jh_250.txt -demand_file $Z"
run "cube250 jh"       $CU250
run "cube250 jh_pace1" $CU250 -mosaic_cold_pace 1
# (e) F7 降速矩阵(与 p23/p23b 的 base 行对照)
for G in 7 2 24 26; do for F in 2 4; do for DIR in 0 1 2; do
  DG="-flex_degrade_gpu $G -flex_degrade_factor $F -flex_degrade_dir $DIR"
  run "g$G f$F d$DIR hostl"   $M -flex_nv_gbps 360 $DG -mosaic_host_lambda
done; done; done
# F7 组合与集合降速
run "g2f4d1 hostl_R"    $M -flex_nv_gbps 360 -flex_degrade_gpu 2 -flex_degrade_factor 4 -flex_degrade_dir 1 -mosaic_host_lambda $R
run "g2f4d1 hostl_pace" $M -flex_nv_gbps 360 -flex_degrade_gpu 2 -flex_degrade_factor 4 -flex_degrade_dir 1 -mosaic_host_lambda -mosaic_cold_pace 1
for SET in 2,10 2,10,18; do
  T=$(echo $SET | tr ',' '_')
  run "relay$T hostl" $M -flex_nv_gbps 360 -flex_degrade_gpus $SET -flex_degrade_factor 4 -flex_degrade_dir 1 -mosaic_host_lambda
done
echo DONE >> $LOG
