#!/bin/bash
# 39_partial_reconf: per-port (partial/async) reconfiguration semantics
# vs the 35-group barrier archive. n=32, k=1, 800G/500ns, 4MB chunks,
# pay-initial (same conventions as 35_optical_reconfig).
#
# Schemes (all executed by `htsim_optical -plan X -async`, i.e. the SAME
# fabric semantics for every column):
#   ron_async   RoN oracle re-billed: every d's plan executed async
#   rot         bidirectional-ring rotation family (port-billing shape)
#   skeleton    persistent-gateway construction (ablation)
#   funnel      chain-funnel construction (ablation, expected negative)
#   harvest_as  Harvest theta-opt plan async (single topology: == barrier)
# MOSAIC-O-PR := argmin over all async candidates (exact-simulation
# selection, same license as MOSAIC-O's scratch evaluation).
# Barrier-side numbers are NOT rerun (frozen archive, see results.csv of
# 35_optical_reconfig).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
GEN=tools/ron_gen.py
PPG=tools/port_plan_gen.py
P3=python3
SP=cache/optical/tmp39
LOG=${LOG:-cache/optical/x39_partial.log}
mkdir -p "$SP"
: > "$LOG"
md5sum $BIN >> "$LOG"

RS="10 100 1000 10000"

dem_file() {
  case $1 in
    uniform) echo cache/optical/demand_n32_uniform.txt;;
    zipf10)  echo cache/nonuniform/demand_n32_zipf1.0_seed42.txt;;
    zipf20)  echo cache/nonuniform/demand_n32_zipf2.0_seed42.txt;;
    persrc)  echo cache/nonuniform/demand_n32_persrc1.0_seed42.txt;;
  esac
}

# uniform as an explicit demand file (rot/skeleton need one)
$P3 - <<'PYEOF'
with open('cache/optical/demand_n32_uniform.txt','w') as f:
    for s in range(32):
        for d in range(32):
            if s != d:
                f.write('demand %d %d 2097152\n' % (s, d))
PYEOF

run_async() { # plan file -> "STATUS T PORTR"
  $BIN -plan "$1" -async | \
    sed -n 's/^optical \(PASSED\|FAILED\) async.*port_reconfigs \([0-9]*\) completion_us \([0-9.e+]*\) .*/\1 \3 \2/p'
}

for DEM in uniform zipf10 zipf20 persrc; do
  DF=$(dem_file $DEM)
  for R in $RS; do
    # ---- ron_async oracle (every d, async billing)
    BEST=; BESTD=
    for D in $(seq 1 31); do
      if [ "$DEM" = uniform ]; then
        $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --reconf-us $R \
          --strategy d$D --output $SP/r.txt 2>/dev/null
      else
        $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --reconf-us $R \
          --strategy d$D --demand-file $DF --output $SP/r.txt 2>/dev/null
      fi
      read ST T PR <<< "$(run_async $SP/r.txt)"
      [ "$ST" = PASSED ] || continue
      if [ -z "$BEST" ] || awk "BEGIN{exit !($T < $BEST)}"; then
        BEST=$T; BESTD=$D
      fi
    done
    echo "ROW $DEM $R ron_async d=$BESTD $BEST" >> "$LOG"

    # ---- rot family
    for G in 1,31 1,5,31 1,7,25,31; do
      $P3 $PPG --demand-file $DF --reconf-us $R --mode rot --gens $G \
        --output $SP/rot.txt 2>/dev/null
      read ST T PR <<< "$(run_async $SP/rot.txt)"
      echo "ROW $DEM $R rot g=$G $T $ST" >> "$LOG"
    done

    # ---- skeleton (ablation)
    for C in 1 2; do for DD in 2 3; do
      $P3 $PPG --demand-file $DF --reconf-us $R --mode skeleton \
        --gateways $C --shifts $DD --output $SP/sk.txt 2>/dev/null
      read ST T PR <<< "$(run_async $SP/sk.txt)"
      echo "ROW $DEM $R skeleton c=${C}d${DD} $T $ST" >> "$LOG"
    done; done

    # ---- funnel (ablation, chain recoupling negative)
    $P3 $PPG --demand-file $DF --reconf-us $R --mode chain \
      --gateways 1 --cold-span 4 --shifts 2 --output $SP/fn.txt 2>/dev/null
    read ST T PR <<< "$(run_async $SP/fn.txt)"
    echo "ROW $DEM $R funnel c1L4d2 $T $ST" >> "$LOG"
  done
done
echo DONE >> "$LOG"
