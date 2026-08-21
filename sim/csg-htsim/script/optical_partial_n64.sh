#!/bin/bash
# 39b: n=64 supplement cells for partial-reconfig (B1 of review round 2):
#   zipf1.0 R=100  (barrier mosaic_o + ron_best barrier + async columns)
#   persrc  R=1000 (same)
# One-layer background, DONE marker (handoff 3.7).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
GEN=tools/ron_gen.py
PPG=tools/port_plan_gen.py
P3=python3
SP=cache/optical/tmp39
LOG=${LOG:-cache/optical/x39b_n64.log}
mkdir -p "$SP"
: > "$LOG"
Z64=cache/nonuniform/demand_n64_zipf1.0_seed42.txt
P64=cache/nonuniform/demand_n64_persrc1.0_seed42.txt
[ -s "$P64" ] || $P3 tools/gen_nonuniform_a2a.py --nodes 64 \
  --message-bytes 2097152 --alpha 1.0 --seed 42 --mode persrc \
  --output "$P64"

run_async() {
  $BIN -plan "$1" -async | \
    sed -n 's/^optical \(PASSED\|FAILED\) async.*port_reconfigs \([0-9]*\) completion_us \([0-9.e+]*\) .*/\1 \3 \2/p'
}
run_bar() {
  $BIN -plan "$1" | \
    sed -n 's/^optical \(PASSED\|FAILED\) .*completion_us \([0-9.e+]*\) .*/\1 \2/p'
}

for CELL in "zipf $Z64 100" "persrc $P64 1000"; do
  set -- $CELL; DEM=$1; DF=$2; R=$3
  # barrier mosaic_o (slow at n=64: or-opt planner)
  MO=$($BIN -mosaic -nodes 64 -demand_file $DF -reconf_us $R)
  echo "ROW $DEM $R mosaic_o_barrier $(echo "$MO" | grep -oE '(PASSED|FAILED)') $(echo "$MO" | grep -oE 'completion_us [0-9.e+]*')" >> "$LOG"
  # ron oracle: barrier + async, full 63-d scan
  BB=; BBD=; BA=; BAD=
  for D in $(seq 1 63); do
    $P3 $GEN --nodes 64 --bytes-per-pair 2097152 --reconf-us $R \
      --strategy d$D --demand-file $DF --output $SP/r64.txt 2>/dev/null
    read ST T <<< "$(run_bar $SP/r64.txt)"
    if [ "$ST" = PASSED ] && { [ -z "$BB" ] || awk "BEGIN{exit !($T < $BB)}"; }; then BB=$T; BBD=$D; fi
    read ST T PR <<< "$(run_async $SP/r64.txt)"
    if [ "$ST" = PASSED ] && { [ -z "$BA" ] || awk "BEGIN{exit !($T < $BA)}"; }; then BA=$T; BAD=$D; fi
  done
  echo "ROW $DEM $R ron_best_barrier d=$BBD $BB" >> "$LOG"
  echo "ROW $DEM $R ron_async d=$BAD $BA" >> "$LOG"
  # rot family async
  for G in 1,63 1,9,63; do
    $P3 $PPG --nodes 64 --demand-file $DF --reconf-us $R --mode rot \
      --gens $G --output $SP/rot64.txt 2>/dev/null
    read ST T PR <<< "$(run_async $SP/rot64.txt)"
    echo "ROW $DEM $R rot g=$G $T $ST" >> "$LOG"
  done
done
echo DONE >> "$LOG"
