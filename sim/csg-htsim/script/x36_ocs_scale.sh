#!/bin/bash
# 36_ocs: OCS weak-scaling grid, n in {256,512,1024}, k=1, 16MB chunks.
# Schemes per (n, R):
#   ron d-grid : RoN plans on a log-spaced d grid; ron_auto = model-min
#                over the grid, ron_best = sim-min (windowed oracle; the
#                full n-1 scan is O(n) sims -- stated, not hidden).
#   mosaic_o   : scale-mode online planner (greedy Birkhoff + pruned
#                Cayley + NN static), wall time = its per-A2A planning.
# Also records ron_gen wall time (its per-A2A construction: byte fill is
# demand-dependent, so the plan must be re-emitted per collective).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
GEN=tools/ron_gen.py
LOG=cache/optical/x36_ocs.log
SP=cache/optical/tmp
: > "$LOG"
echo "binary_md5 $(md5sum $BIN | cut -d' ' -f1)" >> "$LOG"

grid_for() {
  case $1 in
    256)  echo "1 2 4 8 16 32 64 128 255";;
    512)  echo "1 2 4 8 16 32 64 128 256 511";;
    1024) echo "1 2 4 8 16 32 64 128 256 512 1023";;
  esac
}

for N in 256 512 1024; do
  D=cache/nonuniform/demand_n${N}_zipf1.0_seed42.txt
  python3 - <<PYEOF >> "$LOG"
n=$N
dem=[[0]*n for _ in range(n)]
for line in open("$D"):
    p=line.split()
    if len(p)==4 and p[0]=="demand": dem[int(p[1])][int(p[2])]=int(p[3])
rows=[sum(r) for r in dem]
cols=[sum(dem[s][d] for s in range(n)) for d in range(n)]
bw=800e9/8
print("n%d,0,lower_bound,max_line_cut,%.1f"%(n,max(max(rows),max(cols))/bw*1e6))
PYEOF
  for R in 10 1000; do
    BEST=""; BESTD=""; MBEST=""; MBESTD=""; MT=""
    for DD in $(grid_for $N); do
      G0=$(date +%s.%N)
      python3 $GEN --nodes $N --bytes-per-pair 2097152 --reconf-us $R \
        --strategy d$DD --demand-file $D --chunk-bytes 16777216 \
        --output $SP/ron_scale.txt 2>$SP/ron_scale.err
      G1=$(date +%s.%N)
      MC=$(grep -o "model_cost=[0-9.]*" $SP/ron_scale.err | cut -d= -f2)
      T=$(timeout 1200 $BIN -plan $SP/ron_scale.txt | sed -n 's/.*completion_us \([0-9.e+]*\).*/\1/p')
      GT=$(awk "BEGIN{printf \"%.1f\", $G1-$G0}")
      echo "n${N},R${R},ron_d${DD},model=${MC} gen_s=${GT},${T}" >> "$LOG"
      if [ -n "$T" ] && { [ -z "$BEST" ] || awk "BEGIN{exit !($T < $BEST)}"; }; then BEST=$T; BESTD=$DD; fi
      if [ -n "$MC" ] && { [ -z "$MBEST" ] || awk "BEGIN{exit !($MC < $MBEST)}"; }; then MBEST=$MC; MBESTD=$DD; MT=$T; fi
    done
    echo "n${N},R${R},ron_best_grid,d=$BESTD,$BEST" >> "$LOG"
    echo "n${N},R${R},ron_auto_grid,d=$MBESTD,$MT" >> "$LOG"
    echo "n$N R$R ron_best d=$BESTD $BEST | ron_auto d=$MBESTD $MT"

    T0=$(date +%s.%N)
    MO=$(timeout 1200 $BIN -mosaic -nodes $N -demand_file $D -chunk_bytes 16777216 -reconf_us $R)
    T1=$(date +%s.%N)
    T=$(echo "$MO" | sed -n 's/.*completion_us \([0-9.e+]*\).*/\1/p')
    ST=$(echo "$MO" | grep -oE "PASSED|FAILED")
    RC=$(echo "$MO" | grep -oE "reconfigs [0-9]+")
    W=$(awk "BEGIN{printf \"%.0f\", $T1-$T0}")
    echo "n${N},R${R},mosaic_o,$ST $RC wall_s=$W,$T" >> "$LOG"
    echo "n$N R$R mosaic_o $ST $RC wall_s=$W $T"
  done
done
echo DONE >> "$LOG"; echo DONE
