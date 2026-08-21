#!/bin/bash
# 37 组 OCS:switch-centric incast/mixed × R × {ron_auto, ron_best, harvest,
# mosaic_o}。100G/1000ns 对齐 EPS clos 线速(跨侧可比),k=1,4MB chunk。
# 单层后台:nohup script/x37_ocs.sh > cache/optical/x37_ocs.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
GEN=tools/ron_gen.py
P3=python3
GRB=/home/weihaoni/miniconda3/envs/myenv/bin/python3
OUT=${OUT:-cache/optical/x37_ocs.csv}
SP=cache/optical/tmp37
mkdir -p "$SP"
echo "demand,R_us,scheme,detail,completion_us" > "$OUT"
GB=100; DN=1000; CH=4194304
RS="10 1000"
DEMS=${DEMS:-incast1 incast2 incast4 incast8 mixed_incast}

sim_time() { $BIN -plan "$1" | sed -n 's/.*completion_us \([0-9.e+]*\).*/\1/p'; }

for DEM in $DEMS; do
  DF=cache/nonuniform/demand_n32_${DEM}.txt
  LB=$($P3 - <<PYEOF
n=32
dem=[[0]*n for _ in range(n)]
for line in open("$DF"):
    p=line.split()
    if len(p)==4 and p[0]=="demand": dem[int(p[1])][int(p[2])]=int(p[3])
rows=[sum(dem[s]) for s in range(n)]
cols=[sum(dem[s][d] for s in range(n)) for d in range(n)]
print("%.1f"%(max(max(rows),max(cols))/($GB*1e9/8)*1e6))
PYEOF
)
  echo "$DEM,0,lower_bound,max_line_cut,$LB" >> "$OUT"; echo "$DEM LB $LB"

  for R in $RS; do
    $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --gbps $GB --delay-ns $DN \
      --reconf-us $R --strategy auto --demand-file $DF \
      --output $SP/ra.txt 2>$SP/ra.err
    DA=$(grep -o "chose d=[0-9]*" $SP/ra.err | cut -d= -f2)
    T=$(sim_time $SP/ra.txt)
    echo "$DEM,$R,ron_auto,d=$DA,$T" >> "$OUT"; echo "$DEM $R ron_auto d=$DA $T"

    BEST=""; BESTD=""
    for D in $(seq 1 31); do
      $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --gbps $GB --delay-ns $DN \
        --reconf-us $R --strategy d$D --demand-file $DF \
        --output $SP/rb.txt 2>/dev/null || continue
      T=$(sim_time $SP/rb.txt)
      if [ -n "$T" ] && { [ -z "$BEST" ] || awk "BEGIN{exit !($T < $BEST)}"; }; then
        BEST=$T; BESTD=$D
      fi
    done
    echo "$DEM,$R,ron_best,d=$BESTD,$BEST" >> "$OUT"; echo "$DEM $R ron_best d=$BESTD $BEST"

    MO=$(timeout 1200 $BIN -mosaic -nodes 32 -gbps $GB -delay_ns $DN \
         -demand_file $DF -reconf_us $R)
    T=$(echo "$MO" | sed -n 's/.*completion_us \([0-9.e+]*\).*/\1/p')
    RC=$(echo "$MO" | grep -oE "reconfigs [0-9]+")
    ST=$(echo "$MO" | grep -oE "PASSED|FAILED")
    echo "$DEM,$R,mosaic_o,$ST $RC,$T" >> "$OUT"; echo "$DEM $R mosaic_o $ST $RC $T"
  done

  $GRB tools/harvest_synth.py --collective a2a --nodes 32 --degree 1 \
    --demand-file $DF --gbps $GB --delay-ns $DN --chunk-bytes $CH \
    --reconf-us 0 --time-limit 300 --output $SP/hv_$DEM.txt 2>$SP/hv_$DEM.err
  if [ -s $SP/hv_$DEM.txt ]; then
    T0=$(sim_time $SP/hv_$DEM.txt)
    TH=$(grep -o "theta=[0-9.]*" $SP/hv_$DEM.err)
    for R in $RS; do
      T=$(awk "BEGIN{printf \"%.2f\", $T0 + $R}")
      echo "$DEM,$R,harvest,$TH,$T" >> "$OUT"
    done
    echo "$DEM harvest [$TH] base=$T0"
  else
    echo "$DEM,any,harvest,SKIPPED,-" >> "$OUT"; echo "$DEM harvest SKIPPED"
  fi
done

echo DONE >> "$OUT"
echo DONE
