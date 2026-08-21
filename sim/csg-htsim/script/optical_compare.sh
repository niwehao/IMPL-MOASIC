#!/bin/bash
# Optical comparison harness: RoN (best-d oracle scan), Harvest theta-opt
# static, MOSAIC-O, over demand x R grid. n=32, k=1 for RoN/MOSAIC-O
# (Harvest theta-opt uses its own k; run k=1 here for comparability).
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_optical
GEN=tools/ron_gen.py
P3=python3
GRB=/home/weihaoni/miniconda3/envs/myenv/bin/python3
OUT=${OUT:-cache/optical/compare_v1.csv}
SP=${SP:-cache/optical/tmp}
mkdir -p "$SP"
echo "demand,R_us,scheme,detail,completion_us" > "$OUT"

DEMANDS="uniform zipf10 zipf20"
RS="1 10 100 1000"

dem_file() {
  case $1 in
    zipf10) echo cache/nonuniform/demand_n32_zipf1.0_seed42.txt;;
    zipf20) echo cache/nonuniform/demand_n32_zipf2.0_seed42.txt;;
  esac
}

for DEM in $DEMANDS; do
  for R in $RS; do
    # --- RoN: scan every d, keep best simulated time (opponent oracle)
    BEST=""; BESTD=""
    for D in $(seq 1 31); do
      if [ "$DEM" = uniform ]; then
        $P3 $GEN --nodes 32 --bytes-per-pair 2097152 --reconf-us $R \
          --strategy d$D --output $SP/ron.txt 2>/dev/null
      else
        $P3 $GEN --nodes 32 --reconf-us $R --strategy d$D \
          --demand-file $(dem_file $DEM) --output $SP/ron.txt 2>/dev/null
      fi
      T=$($BIN -plan $SP/ron.txt | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
      if [ -n "$T" ] && { [ -z "$BEST" ] || awk "BEGIN{exit !($T < $BEST)}"; }; then
        BEST=$T; BESTD=$D
      fi
    done
    echo "$DEM,$R,ron_best,d=$BESTD,$BEST" >> "$OUT"
    echo "$DEM $R ron_best d=$BESTD $BEST"

    # --- MOSAIC-O
    if [ "$DEM" = uniform ]; then
      MO=$($BIN -mosaic -nodes 32 -uniform_bytes 2097152 -reconf_us $R)
    else
      MO=$($BIN -mosaic -nodes 32 -demand_file $(dem_file $DEM) -reconf_us $R)
    fi
    T=$(echo "$MO" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
    G=$(echo "$MO" | grep -oE "go [0-9]+ stay [0-9]+")
    echo "$DEM,$R,mosaic_o,$G,$T" >> "$OUT"
    echo "$DEM $R mosaic_o [$G] $T"
  done

  # --- Harvest theta-opt static (R-independent single topology; k=1)
  if [ "$DEM" != uniform ]; then
    $GRB tools/harvest_synth.py --collective a2a --nodes 32 --degree 1 \
      --demand-file $(dem_file $DEM) --chunk-bytes 4194304 \
      --time-limit 300 --reconf-us 10 \
      --output $SP/harvest_$DEM.txt 2>$SP/harvest_$DEM.err
    if [ -s $SP/harvest_$DEM.txt ]; then
      T=$($BIN -plan $SP/harvest_$DEM.txt | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
      TH=$(grep -o "theta=[0-9.]*" $SP/harvest_$DEM.err)
      echo "$DEM,any,harvest_static,$TH,$T" >> "$OUT"
      echo "$DEM harvest_static [$TH] $T"
    else
      echo "$DEM,any,harvest_static,SKIPPED,-" >> "$OUT"
    fi
  fi
done
echo DONE >> "$OUT"
echo DONE
