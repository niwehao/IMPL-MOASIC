#!/usr/bin/env bash
# ndv2(16 GPU 异构混布,TE-CCL 独占拓扑)A2A 对比:
# 400/200/100G 混布 + 仅 2 条跨箱链路;TACCL/SyCCL 无法表达此图
# (见 Note/2026-0708-solver-topology-capabilities.md),故只比 basic/static_ecmp/teccl/fast。
# FAST 按 2 箱 x 8 GPU 分组,但机间只有 2 条物理链路,i->i 映射需多跳
# 中继,属 FAST 模型外拓扑(每 GPU 独立 NIC 假设不成立)。
# NONUNIFORM=1:非均匀 A2A;teccl 用 pow2 padded 计划 + fill 截断。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ===================== 可调参数 =====================
NODES="${NODES:-16}"
TOPOLOGY="${TOPOLOGY:-ndv2}"
PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES:-1024}"
MESSAGE_SIZE_BYTES="${MESSAGE_SIZE_BYTES:-2097152}"
CHUNK_SIZE_BYTES="${CHUNK_SIZE_BYTES:-2097152}"
STORE_FORWARD="${STORE_FORWARD:-1}"
SCHEDULERS="${SCHEDULERS:-static_ecmp basic teccl fast}"
FORCE_RESOLVE="${FORCE_RESOLVE:-0}"
NONUNIFORM="${NONUNIFORM:-0}"
# ====================================================

EXTRA_TOPO_ARGS=""
EX="$ROOT/examples/ndv2_16"
NU="$ROOT/cache/nonuniform"
mkdir -p "$EX"
TECCL_TOPO="$EX/ndv2_topo.json"
TECCL_WORKFLOW="$EX/teccl_ndv2_a2a_2mb_workflow.json"

[[ -f "$TECCL_TOPO" ]] || cat > "$TECCL_TOPO" <<'JSON'
{"name": "ndv2", "node_count": 16}
JSON
[[ -f "$TECCL_WORKFLOW" ]] || cat > "$TECCL_WORKFLOW" <<'JSON'
{
  "collective": "alltoall",
  "chunk_size_bytes": 2097152,
  "num_chunks": 1,
  "num_epochs": 32,
  "epoch_multiplier": 8,
  "solution_method": 1,
  "switch_copy": false,
  "switch_to_gpu_link_on": true,
  "time_limit_hours": 0.15
}
JSON

if [[ "$NONUNIFORM" == "1" ]]; then
    nonuniform_setup
    SIZE_TAG="${PAD_MB}mb"
    TECCL_SCHED="$NU/teccl_ndv2_a2a_${SIZE_TAG}.sched"
else
    DEMAND_FILE=""
    SIZE_TAG="2mb"
    TECCL_SCHED="$EX/teccl_ndv2_a2a_${SIZE_TAG}.sched"
fi

htsim_build
echo "== ndv2(16) A2A $([[ "$NONUNIFORM" == "1" ]] && echo "NONUNIFORM($NONUNI_TAG, pad ${PAD_MB}MB)" || echo "${MESSAGE_SIZE_BYTES}B/pair") packet ${PACKET_SIZE_BYTES}B store_forward=$STORE_FORWARD"

run_plan_scheduler() {  # <label> <scheduler_name> <uniform_sched>
    local label="$1" sched_name="$2" uniform_sched="$3"
    if [[ "$NONUNIFORM" == "1" ]]; then
        local filled="${uniform_sched%.sched}_${NONUNI_TAG}_filled.sched"
        if [[ "$FORCE_RESOLVE" == "1" || ! -f "$filled" ]]; then
            fill_schedule "$uniform_sched" "$filled" >/dev/null
        fi
        htsim_run "$label" "$sched_name" "$filled"
    else
        htsim_run "$label" "$sched_name" "$uniform_sched"
    fi
}

for s in $SCHEDULERS; do
    case "$s" in
    static_ecmp|basic)
        htsim_run "$s" "$s" ;;
    fast)
        htsim_run fast fast_schedule ;;
    teccl)
        if [[ "$NONUNIFORM" == "1" ]]; then
            WORKFLOW_PADDED="$NU/teccl_ndv2_a2a_${SIZE_TAG}_workflow.json"
            make_padded_teccl_workflow "$TECCL_WORKFLOW" "$WORKFLOW_PADDED"
            TECCL_WORKFLOW="$WORKFLOW_PADDED"
        fi
        teccl_generate "$TECCL_TOPO" "$TECCL_WORKFLOW" "$TECCL_SCHED" >/dev/null
        run_plan_scheduler teccl teccl_schedule "$TECCL_SCHED" ;;
    *) echo "unknown scheduler: $s" >&2; exit 1 ;;
    esac
done
