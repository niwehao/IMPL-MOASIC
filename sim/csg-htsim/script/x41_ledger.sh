#!/bin/bash
# x41 MOSAIC-L: global resource ledger (-mosaic_ledger) vs the archived
# MOSAIC-U (21-group mainline + x21_v1_robust + 19/14/22 groups).
# Frozen config: C6, band 5%, chain cap 1MB, aggregate rate correction.
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
LOG=${LOG:-cache/nonuniform/x41_ledger.log}
: > "$LOG"
md5sum $BIN >> "$LOG"

run() { # tag, args...
  local TAG=$1; shift
  echo "$TAG $($BIN -nodes 32 -packet_size 1024 \
    -scheduler mosaic_schedule -workload A2A -all_to_all -store_forward \
    -quiet "$@" 2>/dev/null | grep -oE \
    'validation (PASSED|FAILED).*completion_us [0-9.]+|queue_drops [0-9]+|mosaic_corridor_bytes.*' \
    | tr '\n' ' ')" >> "$LOG"
}

Z10=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
RAIL="-topology flex_rail -flex_nic_gbps 180 -flex_taper_gbps 180"
C6="-mosaic_corridors 6"
LED="-mosaic_ledger"

# ---- mainline 8 nv (record; matches inline runs)
for NV in 2400 720 600 480 420 360 180 90; do
  run "main L nv=$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 \
    $C6 $LED -demand_file $Z10
done
# raw (no rate correction) ablation, 8 nv
for NV in 2400 720 600 480 420 360 180 90; do
  run "main RAW nv=$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 \
    $C6 -mosaic_ledger_raw -demand_file $Z10
done

# ---- robustness (archived C6 in x21_v1_robust.log; missing cells get a
#      fresh legacy C6 row for a same-binary comparison)
for SEED in 43 44; do
  DF=cache/nonuniform/demand_n32_zipf1.0_seed${SEED}.txt
  for NV in 2400 360 90; do
    run "rob L seed$SEED nv=$NV" $RAIL -flex_nv_gbps $NV \
      -chunk_size 33554432 $C6 $LED -demand_file $DF
  done
done
run "rob LEG seed43 nv=360" $RAIL -flex_nv_gbps 360 -chunk_size 33554432 \
  $C6 -demand_file cache/nonuniform/demand_n32_zipf1.0_seed43.txt
for A in 0.5 1.5 2.0; do
  DF=cache/nonuniform/demand_n32_zipf${A}_seed42.txt
  case $A in
    0.5) CH=8388608;;
    1.5) CH=67108864;;
    2.0) CH=67108864;;
  esac
  for NV in 2400 360 90; do
    run "rob L zipf$A nv=$NV" $RAIL -flex_nv_gbps $NV -chunk_size $CH \
      $C6 $LED -demand_file $DF
  done
done
run "rob LEG zipf1.5 nv=90" $RAIL -flex_nv_gbps 90 -chunk_size 67108864 \
  $C6 -demand_file cache/nonuniform/demand_n32_zipf1.5_seed42.txt
for NV in 2400 360 90; do
  run "rob L persrc nv=$NV" $RAIL -flex_nv_gbps $NV -chunk_size 33554432 \
    $C6 $LED -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt
done

# ---- uniform (contract: 2MB msg, no demand file)
for NV in 2400 360 90; do
  run "unif L nv=$NV" $RAIL -flex_nv_gbps $NV -chunk_size 2097152 \
    -message_size 2097152 $C6 $LED
done
# uniform + PC composition probe (PC's home turf)
for NV in 2400 360; do
  run "unifPC L nv=$NV" $RAIL -flex_nv_gbps $NV -chunk_size 2097152 \
    -message_size 2097152 $C6 $LED -mosaic_pair_color_auto
done

# ---- H5: uniform x taper (exploration-premium test; archived C4 in 14)
for NV in 2400 360; do
  for TP in 90 60; do
    run "utaper L nv=$NV tp=$TP" -topology flex_rail -flex_nic_gbps 180 \
      -flex_taper_gbps $TP -flex_nv_gbps $NV -chunk_size 2097152 \
      -message_size 2097152 $C6 $LED
    run "utaper LEG nv=$NV tp=$TP" -topology flex_rail -flex_nic_gbps 180 \
      -flex_taper_gbps $TP -flex_nv_gbps $NV -chunk_size 2097152 \
      -message_size 2097152 $C6
  done
done
# nonuniform taper (11-group C6 archived: 8212/8418/8675)
for TP in 90 360; do
  run "ntaper L nv=360 tp=$TP" -topology flex_rail -flex_nic_gbps 180 \
    -flex_taper_gbps $TP -flex_nv_gbps 360 -chunk_size 33554432 \
    $C6 $LED -demand_file $Z10
done

# ---- H4: unannounced NIC degrade (19 group; scheduler NOT told)
for F in 2 4; do
  run "deg L hot x$F nv=2400" $RAIL -flex_nv_gbps 2400 \
    -chunk_size 33554432 $C6 $LED -demand_file $Z10 \
    -flex_degrade_gpu 26 -flex_degrade_factor $F
  run "deg L cold x$F nv=2400" $RAIL -flex_nv_gbps 2400 \
    -chunk_size 33554432 $C6 $LED -demand_file $Z10 \
    -flex_degrade_gpu 7 -flex_degrade_factor $F
  run "deg L hot x$F nv=360" $RAIL -flex_nv_gbps 360 \
    -chunk_size 33554432 $C6 $LED -demand_file $Z10 \
    -flex_degrade_gpu 26 -flex_degrade_factor $F
  run "deg L cold x$F nv=360" $RAIL -flex_nv_gbps 360 \
    -chunk_size 33554432 $C6 $LED -demand_file $Z10 \
    -flex_degrade_gpu 7 -flex_degrade_factor $F
done
# raw ledger on the headline degrade cell (what correction buys)
run "deg RAW hot x4 nv=2400" $RAIL -flex_nv_gbps 2400 \
  -chunk_size 33554432 $C6 -mosaic_ledger_raw -demand_file $Z10 \
  -flex_degrade_gpu 26 -flex_degrade_factor 4

# ---- H6: pod wiring (22 group: nv360 uplink 720, C5; archived 8348)
run "pod L nv=360 up=720" -topology flex_pod -flex_spines 4 \
  -flex_taper_gbps 720 -flex_nic_gbps 180 -flex_nv_gbps 360 \
  -chunk_size 33554432 -mosaic_corridors 5 $LED -demand_file $Z10
run "pod L nv=2400 up=720" -topology flex_pod -flex_spines 4 \
  -flex_taper_gbps 720 -flex_nic_gbps 180 -flex_nv_gbps 2400 \
  -chunk_size 33554432 -mosaic_corridors 5 $LED -demand_file $Z10

# ---- drift structural immunity: stale prior must be bitwise-identical
run "drift L fresh persrc nv=360" $RAIL -flex_nv_gbps 360 \
  -chunk_size 33554432 $C6 $LED \
  -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt
run "drift L stale persrc nv=360" $RAIL -flex_nv_gbps 360 \
  -chunk_size 33554432 $C6 $LED \
  -demand_file cache/nonuniform/demand_n32_persrc1.0_seed42.txt \
  -mosaic_prior_demand_file cache/nonuniform/demand_n32_persrc1.0_seed43.txt

# ---- PC on skew probe (composition safety)
run "pc L zipf nv=90" $RAIL -flex_nv_gbps 90 -chunk_size 33554432 \
  $C6 $LED -demand_file $Z10 -mosaic_pair_color_auto

echo DONE >> "$LOG"
