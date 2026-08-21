#!/bin/bash
# x26: donation/skew exploration matrix (report Note/report_donation_phase.md)
# Variants: base | window(E1) | rewf(E2) | skew2(E3) | skew2+rewf | static |
#           static+skew2(pure-static C) | plan(E4) | plan+static
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
LOG=cache/nonuniform/x26_matrix.log
: > $LOG
BIN=./sim/tests/htsim_smart_regular
COMMON="-nodes 32 -topology flex_rail -packet_size 1024 -scheduler mosaic_schedule -mosaic_corridors 6 -workload A2A -all_to_all -store_forward -quiet"

run() {  # run TAG NV CHUNK DEMAND EXTRA...
  local TAG=$1 NV=$2 CHUNK=$3 DEM=$4; shift 4
  local R
  R=$($BIN $COMMON -flex_nv_gbps $NV -chunk_size $CHUNK ${DEM:+-demand_file $DEM} "$@" 2>&1 \
      | grep -oE "validation (PASSED|FAILED) rounds [0-9]+ completion_us [0-9.]+|queue_drops [0-9]+" \
      | tr '\n' ' ')
  echo "TAG $TAG nv $NV $R" >> $LOG
}

variant_flags() {
  case $1 in
    base)   echo "";;
    window) echo "-mosaic_donate_window";;
    rewf)   echo "-mosaic_donate_rewaterfill";;
    skew2)  echo "-mosaic_skew_epoch 2";;
    skewrf) echo "-mosaic_skew_epoch 2 -mosaic_donate_rewaterfill";;
    static) echo "-mosaic_static";;
    statsk) echo "-mosaic_static -mosaic_skew_epoch 2";;
    plan)   echo "-mosaic_phase_plan";;
    planst) echo "-mosaic_phase_plan -mosaic_static";;
  esac
}

D10=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# main grid
for NV in 90 180 360 2400; do
  for V in base window rewf skew2 skewrf static statsk plan planst; do
    run "zipf1.0_seed42_$V" $NV 33554432 $D10 $(variant_flags $V)
  done
done
# robustness: seeds
for SD in 43 44; do
  for NV in 90 360; do
    for V in base skew2 rewf skewrf; do
      run "zipf1.0_seed${SD}_$V" $NV 33554432 cache/nonuniform/demand_n32_zipf1.0_seed${SD}.txt $(variant_flags $V)
    done
  done
done
# robustness: alpha
for AL in 1.5 2.0; do
  for NV in 90 360; do
    for V in base skew2 rewf skewrf; do
      run "zipf${AL}_seed42_$V" $NV 67108864 cache/nonuniform/demand_n32_zipf${AL}_seed42.txt $(variant_flags $V)
    done
  done
done
# robustness: khot2 / persrc @360
for V in base skew2 rewf skewrf; do
  run "khot2_seed42_$V" 360 16777216 cache/nonuniform/demand_n32_khot2_seed42.txt $(variant_flags $V)
  run "persrc1.0_seed42_$V" 360 33554432 cache/nonuniform/demand_n32_persrc1.0_seed42.txt $(variant_flags $V)
done
# degrade straggler @2400 (19组 setup: gpu26 nic/4)
for V in base skew2 rewf skewrf; do
  run "zipf1.0_seed42_degr26x4_$V" 2400 33554432 $D10 -flex_degrade_gpu 26 -flex_degrade_factor 4 $(variant_flags $V)
done
# uniform no-harm (contract: message=chunk=2MB, no demand file)
for NV in 360 2400; do
  for V in base skewrf; do
    run "uniform_$V" $NV 2097152 "" -message_size 2097152 $(variant_flags $V)
  done
done
echo DONE >> $LOG
