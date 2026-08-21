#!/bin/bash
# 36_eps addendum: FAST overflows the default 1GB fabric buffer at scale
# (no retransmit -> FAILED). Opponent-favorable accommodation: raise
# -flex_fabric_buffer_mb until PASSED; one mosaic control at the same
# buffer shows the knob is neutral for us.
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
LOG=cache/nonuniform/x36_fastbuf.log
: > "$LOG"
run() {
  local tag="$1"; shift
  local t0=$(date +%s.%N)
  local out; out=$(timeout 1200 "$BIN" "$@" 2>&1)
  local t1=$(date +%s.%N)
  local v=$(echo "$out" | grep -E "^validation" | head -1)
  local status=$(echo "$v" | grep -o "PASSED\|FAILED" | head -1)
  local us=$(echo "$v" | sed -n 's/.*completion_us \([0-9.]*\).*/\1/p')
  local drops=$(echo "$v" | sed -n 's/.*queue_drops \([0-9]*\).*/\1/p')
  local plan=$(echo "$out" | grep -oE "plan_us [0-9.]+" | head -1 | cut -d' ' -f2)
  echo "$tag $status $us drops=$drops plan_us=${plan:-0} wall_s=$(awk "BEGIN{printf \"%.0f\", $t1-$t0}")" | tee -a "$LOG"
}
for N in 256 512 1024; do
  H=$((N/8)); D=cache/nonuniform/demand_n${N}_zipf1.0_seed42.txt
  C="-nodes $N -topology flex_rail -flex_hosts $H -flex_gpus_per_host 8 \
     -flex_nv_gbps 360 -packet_size 1024 -chunk_size 16777216 \
     -workload A2A -all_to_all -store_forward -quiet -demand_file $D"
  for MB in 4096 16384; do
    run n${N}_fast_buf${MB} $C -scheduler fast_schedule -fast_alpha_beta_guard -flex_fabric_buffer_mb $MB
    run n${N}_ecmp_buf${MB} $C -scheduler static_ecmp -flex_fabric_buffer_mb $MB
    if grep -q "n${N}_fast_buf${MB} PASSED" "$LOG"; then
      run n${N}_mosaic_pc_buf${MB} $C -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_pair_color_auto -flex_fabric_buffer_mb $MB
      break
    fi
  done
done
echo DONE >> "$LOG"; echo DONE
