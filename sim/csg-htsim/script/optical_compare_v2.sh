#!/bin/bash
# 35_optical: full three-way reconfiguration comparison on the optical
# circuit-switched scale-up domain (n=32, k=1, 800G, 500ns, 4MB chunks).
# Schemes:
#   ron_auto   Reconfigure-or-Not, its own model-selected d (deployable)
#   ron_best   RoN oracle: every d simulated, min taken (opponent-best)
#   harvest    Harvest theta-optimal static topology (Gurobi MILP, k=1)
#   mosaic_o   MOSAIC-O online receding-horizon planner
# Drift block: prior = zipf1.0 seed44, actual = seed42 (Harvest resolves
# on the prior; RoN's round structure is demand-oblivious hence immune;
# MOSAIC-O plans online from actual residuals).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
GEN=tools/ron_gen.py
P3=python3
GRB=/home/weihaoni/miniconda3/envs/myenv/bin/python3
OUT=${OUT:-cache/optical/compare_v2.csv}
SP=cache/optical/tmp
mkdir -p "$SP"
echo "demand,R_us,scheme,detail,completion_us" > "$OUT"

RS="1 10 100 1000 10000"

dem_file() {
  case $1 in
    zipf10)  echo cache/nonuniform/demand_n32_zipf1.0_seed42.txt;;
    zipf20)  echo cache/nonuniform/demand_n32_zipf2.0_seed42.txt;;
    persrc)  echo cache/nonuniform/demand_n32_persrc1.0_seed42.txt;;
  esac
}

sim_time() { $BIN -plan "$1" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p'; }

for DEM in uniform zipf10 zipf20 persrc; do
  # ---- lower bound (max line cut + one mandatory configuration R)
  if [ "$DEM" = uniform ]; then
    LB=$($P3 - <<PYEOF
bw=800e9/8
print("%.1f"%( (31*2097152)/bw*1e6 ))
PYEOF
)
  else
    LB=$($P3 - <<PYEOF
f="$(dem_file $DEM)"
n=32
dem=[[0]*n for _ in range(n)]
for line in open(f):
    p=line.split()
    if len(p)==4 and p[0]=="demand": dem[int(p[1])][int(p[2])]=int(p[3])
rows=[sum(dem[s]) for s in range(n)]
cols=[sum(dem[s][d] for s in range(n)) for d in range(n)]
bw=800e9/8
print("%.1f"%(max(max(rows),max(cols))/bw*1e6))
PYEOF
)
  fi
  echo "$DEM,0,lower_bound,max_line_cut,$LB" >> "$OUT"
  echo "$DEM LB $LB"

  for R in $RS; do
    # ---- ron_auto (model d; base bytes 2MB so the model matches scale)
    if [ "$DEM" = uniform ]; then
      $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --reconf-us $R \
        --strategy auto --output $SP/ra.txt 2>$SP/ra.err
    else
      $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --reconf-us $R \
        --strategy auto --demand-file $(dem_file $DEM) \
        --output $SP/ra.txt 2>$SP/ra.err
    fi
    DA=$(grep -o "chose d=[0-9]*" $SP/ra.err | cut -d= -f2)
    T=$(sim_time $SP/ra.txt)
    echo "$DEM,$R,ron_auto,d=$DA,$T" >> "$OUT"; echo "$DEM $R ron_auto d=$DA $T"

    # ---- ron_best (oracle scan)
    BEST=""; BESTD=""
    for D in $(seq 1 31); do
      if [ "$DEM" = uniform ]; then
        $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --reconf-us $R \
          --strategy d$D --output $SP/rb.txt 2>/dev/null
      else
        $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --reconf-us $R \
          --strategy d$D --demand-file $(dem_file $DEM) \
          --output $SP/rb.txt 2>/dev/null
      fi
      T=$(sim_time $SP/rb.txt)
      if [ -n "$T" ] && { [ -z "$BEST" ] || awk "BEGIN{exit !($T < $BEST)}"; }; then
        BEST=$T; BESTD=$D
      fi
    done
    echo "$DEM,$R,ron_best,d=$BESTD,$BEST" >> "$OUT"; echo "$DEM $R ron_best d=$BESTD $BEST"

    # ---- mosaic_o
    if [ "$DEM" = uniform ]; then
      MO=$($BIN -mosaic -nodes 32 -uniform_bytes 2097152 -reconf_us $R)
    else
      MO=$($BIN -mosaic -nodes 32 -demand_file $(dem_file $DEM) -reconf_us $R)
    fi
    T=$(echo "$MO" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
    RC=$(echo "$MO" | grep -oE "reconfigs [0-9]+")
    ST=$(echo "$MO" | grep -oE "PASSED|FAILED")
    echo "$DEM,$R,mosaic_o,$ST $RC,$T" >> "$OUT"; echo "$DEM $R mosaic_o $ST $RC $T"
  done

  # ---- harvest theta-opt static (R-dependent only through the single
  # initial R; sim executes with reconf_us=10 in plan header -> patch R
  # per row instead: completion = sim_time_with_R0 + R. We emit with
  # reconf_us=0 and add R in post: pay_initial makes it exact.)
  if [ "$DEM" != uniform ]; then
    $GRB tools/harvest_synth.py --collective a2a --nodes 32 --degree 1 \
      --demand-file $(dem_file $DEM) --chunk-bytes 4194304 \
      --reconf-us 0 --time-limit 300 \
      --output $SP/hv_$DEM.txt 2>$SP/hv_$DEM.err
    if [ -s $SP/hv_$DEM.txt ]; then
      T0=$(sim_time $SP/hv_$DEM.txt)
      TH=$(grep -o "theta=[0-9.]*" $SP/hv_$DEM.err)
      for R in $RS; do
        T=$(awk "BEGIN{printf \"%.2f\", $T0 + $R}")
        echo "$DEM,$R,harvest,$TH,$T" >> "$OUT"
      done
      echo "$DEM harvest [$TH] base=$T0"
    else
      echo "$DEM,any,harvest,SKIPPED,-" >> "$OUT"
    fi
  else
    # uniform: theta-opt k=1 is any permutation... a ring; equals ron d=1.
    echo "$DEM,any,harvest,degenerates_to_static_ring,-" >> "$OUT"
  fi
done

# ---- drift block: prior seed44 -> actual seed42 (zipf1.0)
PRIOR=cache/nonuniform/demand_n32_zipf1.0_seed44.txt
ACTUAL=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
$GRB tools/harvest_synth.py --collective a2a --nodes 32 --degree 1 \
  --demand-file $PRIOR --actual-file $ACTUAL --chunk-bytes 4194304 \
  --reconf-us 0 --time-limit 300 --output $SP/hv_drift.txt 2>$SP/hv_drift.err
if [ -s $SP/hv_drift.txt ]; then
  T0=$(sim_time $SP/hv_drift.txt)
  for R in $RS; do
    T=$(awk "BEGIN{printf \"%.2f\", $T0 + $R}")
    echo "zipf10_drift,$R,harvest,prior=seed44,$T" >> "$OUT"
  done
  echo "drift harvest base=$T0"
fi
# mosaic_o is online: identical to zipf10 rows (planning input IS the
# actual residual); ron structure is demand-oblivious: identical rows.
echo "zipf10_drift,-,mosaic_o,same_as_zipf10_online,-" >> "$OUT"
echo "zipf10_drift,-,ron,structure_demand_oblivious,-" >> "$OUT"

echo DONE >> "$OUT"
echo DONE
