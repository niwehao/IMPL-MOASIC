#!/usr/bin/env bash
# Shared helpers for per-topology experiment scripts. Source this file.
# Contract: caller sets NODES / TOPOLOGY / PACKET_SIZE_BYTES /
# MESSAGE_SIZE_BYTES / CHUNK_SIZE_BYTES / STORE_FORWARD / EXTRA_TOPO_ARGS
# before calling htsim_run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/sim/tests/htsim_smart_regular"
CONDA="${CONDA:-/home/weihaoni/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-myenv}"

htsim_build() {
    make -C "$ROOT/sim/tests" htsim_smart_regular >/dev/null
}

# htsim_run <label> <scheduler> [schedule_file]
# Prints one summary line; returns non-zero if validation failed.
# When DEMAND_FILE is set (non-uniform mode) it is passed to htsim; the
# plan-based schedulers get the demand through their filled schedule file
# instead, so it is only forwarded for workflow/fast schedulers.
htsim_run() {
    local label="$1" scheduler="$2" sched_file="${3:-}"
    local args=(
        -nodes "$NODES" -topology "$TOPOLOGY"
        -packet_size "$PACKET_SIZE_BYTES"
        -message_size "$MESSAGE_SIZE_BYTES"
        -chunk_size "$CHUNK_SIZE_BYTES"
        -scheduler "$scheduler"
        -workload A2A -all_to_all -quiet
    )
    if [[ "${STORE_FORWARD:-1}" != "0" ]]; then
        args+=(-store_forward)
    fi
    if [[ -n "$sched_file" ]]; then
        args+=(-schedule_file "$sched_file")
    elif [[ -n "${DEMAND_FILE:-}" ]]; then
        args+=(-demand_file "$DEMAND_FILE")
    fi
    # shellcheck disable=SC2206
    args+=(${EXTRA_TOPO_ARGS:-})

    local out
    out="$("$BIN" "${args[@]}" 2>&1 | grep -E "^validation ")"
    local status completion
    status="$(grep -oE "validation [A-Z]+" <<<"$out" | awk '{print $2}')"
    completion="$(grep -oE "completion_us [0-9.]+" <<<"$out" | awk '{print $2}')"
    printf "%-14s %-10s completion_us %-12s\n" "$label" "$status" "$completion"
    [[ "$status" == "PASSED" ]]
}

# teccl_generate <topo_json> <workflow_json> <out_sched>
# The generator caches by content hash: identical inputs hit the cache.
teccl_generate() {
    local topo="$1" workflow="$2" out="$3"
    python3 "$ROOT/tools/teccl_generate_schedule.py" \
        --topo "$topo" --workflow "$workflow" --output "$out"
}

# skip_if_cached <file> -> returns 0 (skip) if the schedule already exists
skip_if_cached() {
    [[ -f "$1" ]]
}

# ---------------------------------------------------------------------------
# Non-uniform A2A support. NONUNIFORM=1 turns every scheduler case into:
#   demand matrix (seeded hot-dst zipf, same total volume as uniform)
#   -> uniform-only solvers are re-solved at PAD_BYTES = pow2(max pair)
#   -> fill tool truncates the padded plan to the real per-pair bytes
#   -> ecmp/basic/fast consume the demand matrix directly.
# ---------------------------------------------------------------------------

# nonuniform_setup: generates/caches the demand file and exports
# DEMAND_FILE, PAD_BYTES, PAD_MB, NONUNI_TAG; forces CHUNK_SIZE_BYTES=PAD.
nonuniform_setup() {
    local alpha="${NONUNI_ALPHA:-1.0}" seed="${NONUNI_SEED:-42}"
    NONUNI_TAG="zipf${alpha}_seed${seed}"
    DEMAND_FILE="$ROOT/cache/nonuniform/demand_n${NODES}_${NONUNI_TAG}.txt"
    mkdir -p "$ROOT/cache/nonuniform"
    if [[ ! -f "$DEMAND_FILE" ]]; then
        python3 "$ROOT/tools/gen_nonuniform_a2a.py" --nodes "$NODES" \
            --message-bytes "$MESSAGE_SIZE_BYTES" --alpha "$alpha" \
            --seed "$seed" --output "$DEMAND_FILE"
    fi
    PAD_BYTES="$(awk '/padded_pow2_bytes/{print $5; exit}' "$DEMAND_FILE")"
    PAD_MB=$((PAD_BYTES / 1048576))
    CHUNK_SIZE_BYTES="$PAD_BYTES"
    echo "nonuniform demand $DEMAND_FILE pad ${PAD_MB}MB"
}

# make_padded_teccl_workflow <base_workflow_json> <out_json>
make_padded_teccl_workflow() {
    python3 - "$1" "$2" "$PAD_BYTES" <<'PYEOF'
import json, sys
w = json.load(open(sys.argv[1]))
w['chunk_size_bytes'] = int(sys.argv[3])
json.dump(w, open(sys.argv[2], 'w'), indent=2)
PYEOF
}

# make_padded_taccl_topo <base_2mb_topo_json> <out_json>
# Re-expresses the 2MB-calibrated profile at PAD_BYTES per chunk by
# RESCALING THE TIME UNIT instead of multiplying betas: betas/invbws keep
# their 2MB numbers (one unit now means factor x as much wall time) and
# alphas shrink by the factor. Numerically equivalent to beta*factor, but
# keeps the makespan inside TACCL's hardcoded time sentinels
# (routing ST=5e6, scheduler ST=5e5), which a beta*16 encoding provably
# violates (routing/scheduler MIP -> "Infeasible model", 2026-07-10).
make_padded_taccl_topo() {
    python3 - "$1" "$2" "$PAD_BYTES" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1]))
factor = int(sys.argv[3]) / 2097152.0
alpha_beta = all(v == 0 or v == t['alpha'] + t['betas'][i][j]
                 for i, row in enumerate(t['invbws']) for j, v in enumerate(row))
t['name'] = t.get('name', '') + '_pad'
# Round rescaled alphas to integers: fractional coefficients with TACCL's
# 1e-9 feasibility/integrality tolerances stall the routing MIP (ndv2_32:
# 600s zero incumbents vs 276s OPTIMAL on the integral 2MB model). The
# rounding error (<0.1us) is negligible against multi-ms betas.
old_alpha = t['alpha']
t['alpha'] = int(round(old_alpha / factor))
if alpha_beta:
    t['invbws'] = [[(t['alpha'] + t['betas'][i][j]) if v != 0 else 0
                    for j, v in enumerate(row)]
                   for i, row in enumerate(t['invbws'])]
if 'remote_alpha' in t:
    old_remote = t['remote_alpha']
    t['remote_alpha'] = int(round(old_remote / factor))
    if t.get('remote_invbw'):
        t['remote_invbw'] = t['remote_invbw'] - (old_remote - t['remote_alpha'])
json.dump(t, open(sys.argv[2], 'w'), indent=2)
PYEOF
}

# make_padded_syccl_config <base_2mb_config_json> <out_json> <result_json_name>
make_padded_syccl_config() {
    python3 - "$1" "$2" "$PAD_BYTES" "$3" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
c['coll']['byte'] = int(sys.argv[3])
def fix(d):
    for k, v in d.items():
        if isinstance(v, dict):
            fix(v)
        elif k == 'solve_output' and isinstance(v, str):
            d[k] = './' + sys.argv[4]
fix(c)
json.dump(c, open(sys.argv[2], 'w'), indent=2)
PYEOF
}

# fill_schedule <uniform_padded_sched> <out_sched>
fill_schedule() {
    python3 "$ROOT/tools/fill_schedule_nonuniform.py" \
        --schedule "$1" --demand "$DEMAND_FILE" --output "$2"
}
