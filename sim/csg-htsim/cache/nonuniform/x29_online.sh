#!/bin/bash
# x29: flex_cube online-scheme comparison batch.
# Phases: A 主线混布全方案+corridor 消融; B rot 扫描; C 速率画像;
#         D 退化对拍 vs flex_rail; E uniform demand.
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x29_online.log
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet"
NONUNI="-chunk_size 33554432 -demand_file $DZ"
UNI="-chunk_size 2097152 -message_size 2097152"
R_FLAGS="-mosaic_ramp_cold 1 -mosaic_ramp_event -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill"

run() { # tag, then full args
  local tag="$1"; shift
  local out
  out=$($BIN $COMMON "$@" 2>&1 | grep -E "^validation" | head -1 \
        | sed 's/ flows.*queue_drops/ drops/;s/ split.*$//')
  echo "$tag $out" >> $LOG
}

: > $LOG
# ---- A: mainline profile, zipf1.0/seed42 ----
P="-topology flex_cube"
run A_ecmp        $P $NONUNI -scheduler static_ecmp
run A_basic       $P $NONUNI -scheduler basic
run A_fast        $P $NONUNI -scheduler fast_schedule -fast_alpha_beta_guard
for C in 1 2 3 4 5 6; do
  run A_mosaic_c$C $P $NONUNI -scheduler mosaic_schedule -mosaic_corridors $C
done
run A_mosaic_r    $P $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6 $R_FLAGS
run A_mosaic_coldE1 $P $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_ramp_cold 1 -mosaic_ramp_event
run A_mosaic_static $P $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_static
run A_mosaic_uninit $P $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_uniform_init

# ---- B: rot sweep (which B_in the hot dst gets) ----
for ROT in 4 8 12; do
  PB="-topology flex_cube -flex_cube_mix_rot $ROT"
  run B_rot${ROT}_ecmp   $PB $NONUNI -scheduler static_ecmp
  run B_rot${ROT}_fast   $PB $NONUNI -scheduler fast_schedule -fast_alpha_beta_guard
  run B_rot${ROT}_mosaic $PB $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6
  run B_rot${ROT}_mosaic_r $PB $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6 $R_FLAGS
done

# ---- C: speed profiles (mesh faster AND slower than the 180G NIC) ----
for SPEC in 250x16 400x16 100x16 50x16; do
  PC="-topology flex_cube -flex_cube_speeds ${SPEC}"
  run C_${SPEC}_ecmp   $PC $NONUNI -scheduler static_ecmp
  run C_${SPEC}_fast   $PC $NONUNI -scheduler fast_schedule -fast_alpha_beta_guard
  run C_${SPEC}_mosaic $PC $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6
  run C_${SPEC}_mosaic_r $PC $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6 $R_FLAGS
done

# ---- D: degenerate full-crossbar vs flex_rail ----
for V in 50 300; do
  NV=$((7 * V))
  PD="-topology flex_cube -flex_cube_full_gbps $V"
  PR="-topology flex_rail -flex_nv_gbps $NV"
  run D_full${V}_ecmp    $PD $NONUNI -scheduler static_ecmp
  run D_rail${NV}_ecmp   $PR $NONUNI -scheduler static_ecmp
  run D_full${V}_mosaic  $PD $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6
  run D_rail${NV}_mosaic $PR $NONUNI -scheduler mosaic_schedule -mosaic_corridors 6
  run D_full${V}_fast    $PD $NONUNI -scheduler fast_schedule -fast_alpha_beta_guard
  run D_rail${NV}_fast   $PR $NONUNI -scheduler fast_schedule -fast_alpha_beta_guard
done

# ---- E: uniform A2A on mainline profile ----
run E_uni_ecmp   -topology flex_cube $UNI -scheduler static_ecmp
run E_uni_basic  -topology flex_cube $UNI -scheduler basic
run E_uni_fast   -topology flex_cube $UNI -scheduler fast_schedule -fast_alpha_beta_guard
run E_uni_mosaic -topology flex_cube $UNI -scheduler mosaic_schedule -mosaic_corridors 6
run E_uni_mosaic_r -topology flex_cube $UNI -scheduler mosaic_schedule -mosaic_corridors 6 $R_FLAGS

echo DONE >> $LOG
