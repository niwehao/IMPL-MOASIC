#!/usr/bin/env bash
# x45: MOSAIC-O-S (-spectrum) full grid + drift + n64.
# Opponents are FROZEN archives (35 barrier best / 39 PR / 40-43 O-W/O-A),
# none rerun; this script only runs the new scheme.
# Output: $LOG (raw lines) and stdout progress; results.csv is built by
# tools/x45_results.py from $LOG.
set -u
cd "$(dirname "$0")/.."
BIN=${BIN:-./sim/tests/htsim_optical}
NU=cache/nonuniform
OPT=cache/optical
LOG=${LOG:-cache/optical/x45_spectrum.log}

{
echo "# x45 spectrum grid  $(date +%F_%T)"
echo "# binary_md5 $(md5sum $BIN | cut -d' ' -f1)"

for dm in uniform zipf10 zipf20 persrc; do
  case $dm in
    uniform) DF="$OPT/demand_n32_uniform.txt"; TH="";;
    zipf10)  DF="$NU/demand_n32_zipf1.0_seed42.txt"; TH="-static_order_file $OPT/theta_order_zipf1.0.txt";;
    zipf20)  DF="$NU/demand_n32_zipf2.0_seed42.txt"; TH="-static_order_file $OPT/theta_order_zipf2.0_v2.txt";;
    persrc)  DF="$NU/demand_n32_persrc1.0_seed42.txt"; TH="-static_order_file $OPT/theta_order_persrc_v2.txt";;
  esac
  for r in 10 100 1000 10000; do
    echo "CELL $dm $r $($BIN -spectrum -nodes 32 -demand_file $DF -reconf_us $r $TH 2>/dev/null | grep ' spectrum ')"
  done
done

for t in 3000 5000 8000 9500; do
  echo "DRIFT $t $($BIN -spectrum -nodes 32 -demand_file $NU/demand_n32_zipf1.0_seed42.txt -reconf_us 1000 -static_order_file $OPT/theta_order_zipf1.0.txt -drift_file $OPT/drift_half_seed44.txt -drift_us $t 2>/dev/null | grep ' spectrum ')"
done

echo "N64 zipf10_1000 $($BIN -spectrum -nodes 64 -demand_file $NU/demand_n64_zipf1.0_seed42.txt -reconf_us 1000 2>/dev/null | grep ' spectrum ')"
echo "N64 zipf10_100 $($BIN -spectrum -nodes 64 -demand_file $NU/demand_n64_zipf1.0_seed42.txt -reconf_us 100 2>/dev/null | grep ' spectrum ')"
echo "N64 persrc_1000 $($BIN -spectrum -nodes 64 -demand_file $NU/demand_n64_persrc1.0_seed42.txt -reconf_us 1000 2>/dev/null | grep ' spectrum ')"

echo "DONE"
} | tee "$LOG"
