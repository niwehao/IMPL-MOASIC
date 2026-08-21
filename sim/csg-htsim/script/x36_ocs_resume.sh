#!/bin/bash
# 36_ocs resume: remaining cells after the mid-run script edit forced a
# restart -- n512 R10 mosaic_o; n512 R1000 full; n1024 both R (20-min
# caps; abandon-over-cap is policy). Appends to the same log.
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
GEN=tools/ron_gen.py
LOG=cache/optical/x36_ocs.log
SP=cache/optical/tmp

ron_cell() {
  local N=$1 R=$2 GRID="$3"
  local D=cache/nonuniform/demand_n${N}_zipf1.0_seed42.txt
  local BEST="" BESTD="" MBEST="" MBESTD="" MT=""
  for DD in $GRID; do
    local G0=$(date +%s.%N)
    python3 $GEN --nodes $N --bytes-per-pair 2097152 --reconf-us $R \
      --strategy d$DD --demand-file $D --chunk-bytes 16777216 \
      --output $SP/ron_scale.txt 2>$SP/ron_scale.err
    local G1=$(date +%s.%N)
    local MC=$(grep -o "model_cost=[0-9.]*" $SP/ron_scale.err | cut -d= -f2)
    local T=$(timeout 1200 $BIN -plan $SP/ron_scale.txt | sed -n 's/.*completion_us \([0-9.e+]*\).*/\1/p')
    T=$(python3 -c "print('%.1f'%float('$T'))" 2>/dev/null)
    local GT=$(awk "BEGIN{printf \"%.1f\", $G1-$G0}")
    echo "n${N},R${R},ron_d${DD},model=${MC} gen_s=${GT},${T}" >> "$LOG"
    if [ -n "$T" ] && { [ -z "$BEST" ] || awk "BEGIN{exit !($T < $BEST)}"; }; then BEST=$T; BESTD=$DD; fi
    if [ -n "$MC" ] && { [ -z "$MBEST" ] || awk "BEGIN{exit !($MC < $MBEST)}"; }; then MBEST=$MC; MBESTD=$DD; MT=$T; fi
  done
  echo "n${N},R${R},ron_best_grid,d=$BESTD,$BEST" >> "$LOG"
  echo "n${N},R${R},ron_auto_grid,d=$MBESTD,$MT" >> "$LOG"
  echo "n$N R$R ron_best d=$BESTD $BEST | auto d=$MBESTD $MT"
}

mo_cell() {
  local N=$1 R=$2
  local D=cache/nonuniform/demand_n${N}_zipf1.0_seed42.txt
  local T0=$(date +%s.%N)
  local MO=$(timeout 1200 $BIN -mosaic -nodes $N -demand_file $D -chunk_bytes 16777216 -reconf_us $R)
  local T1=$(date +%s.%N)
  local T=$(echo "$MO" | sed -n 's/.*completion_us \([0-9.e+]*\).*/\1/p')
  T=$(python3 -c "print('%.1f'%float('$T'))" 2>/dev/null)
  local ST=$(echo "$MO" | grep -oE "PASSED|FAILED")
  local RC=$(echo "$MO" | grep -oE "reconfigs [0-9]+")
  local W=$(awk "BEGIN{printf \"%.0f\", $T1-$T0}")
  echo "n${N},R${R},mosaic_o,${ST:-TIMEOUT} $RC wall_s=$W,$T" >> "$LOG"
  echo "n$N R$R mosaic_o ${ST:-TIMEOUT} $RC wall_s=$W $T"
}

mo_cell 512 10
ron_cell 512 1000 "8 16 32 64 128 256 511"
mo_cell 512 1000
# lower bound row for n1024
python3 - <<'PYEOF' >> "$LOG"
n=1024
dem=[[0]*n for _ in range(n)]
for line in open("cache/nonuniform/demand_n1024_zipf1.0_seed42.txt"):
    p=line.split()
    if len(p)==4 and p[0]=="demand": dem[int(p[1])][int(p[2])]=int(p[3])
rows=[sum(r) for r in dem]
cols=[sum(dem[s][d] for s in range(n)) for d in range(n)]
print("n1024,0,lower_bound,max_line_cut,%.1f"%(max(max(rows),max(cols))/1e11*1e6))
PYEOF
ron_cell 1024 10 "16 32 64 128 256 512 1023"
mo_cell 1024 10
ron_cell 1024 1000 "16 32 64 128 256 512 1023"
mo_cell 1024 1000
echo DONE >> "$LOG"; echo DONE
