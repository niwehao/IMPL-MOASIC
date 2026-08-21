#!/usr/bin/env bash
# regular_32(32 节点 degree-4 circulant 纯直连图)A2A 对比:
# 编译 -> schedule 生成(带缓存)-> htsim 模拟。
# SyCCL 不支持此拓扑(见 Note/solver_topology_capabilities.md);
# TACCL 用"单主机 32 GPU + 机内 links 矩阵"编码(可解性见能力文档)。
# FAST 在此退化为纯 Birkhoff 置换调度(gpus_per_server=1),且置换目标常
# 非邻居 → 多跳 store-forward,属 FAST 模型外拓扑(见能力文档)。
# NONUNIFORM=1:非均匀 A2A;teccl/taccl 用 pow2 padded 计划 + fill 截断。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ===================== 可调参数 =====================
NODES="${NODES:-32}"
DEGREE="${DEGREE:-4}"
TOPOLOGY="${TOPOLOGY:-regular}"
PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES:-1024}"
MESSAGE_SIZE_BYTES="${MESSAGE_SIZE_BYTES:-2097152}"
CHUNK_SIZE_BYTES="${CHUNK_SIZE_BYTES:-2097152}"
STORE_FORWARD="${STORE_FORWARD:-1}"
LINK_DELAY_US="${LINK_DELAY_US:-5}"
SCHEDULERS="${SCHEDULERS:-static_ecmp basic teccl taccl fast}"   # taccl 单节点编码已验证可解(routing 14s OPTIMAL)
FORCE_RESOLVE="${FORCE_RESOLVE:-0}"
NONUNIFORM="${NONUNIFORM:-0}"
export TACCL_SCHEDULER_TIME_LIMIT="${TACCL_SCHEDULER_TIME_LIMIT:-600}"
export TACCL_ROUTING_TIME_LIMIT="${TACCL_ROUTING_TIME_LIMIT:-600}"
# ====================================================

EXTRA_TOPO_ARGS="-degree $DEGREE -link_delay_us $LINK_DELAY_US"
EX="$ROOT/examples/regular_32"
NU="$ROOT/cache/nonuniform"

if [[ "$NONUNIFORM" == "1" ]]; then
    nonuniform_setup
    SIZE_TAG="${PAD_MB}mb"
else
    DEMAND_FILE=""
    SIZE_TAG="2mb"
fi

TECCL_WORKFLOW="$EX/teccl_regular_32_a2a_2mb_workflow.json"
TECCL_SCHED="$EX/teccl_regular_32_a2a_${SIZE_TAG}.sched"
TACCL_TOPO="$EX/taccl_regular_32_topo_2mb.json"
TACCL_DIR="$ROOT/cache/taccl_regular_32"
TACCL_SCHED="$TACCL_DIR/taccl_regular_32_a2a_${SIZE_TAG}.sched"
if [[ "$NONUNIFORM" == "1" ]]; then
    TECCL_SCHED="$NU/teccl_regular_32_a2a_${SIZE_TAG}.sched"
    TACCL_DIR="$NU/taccl_regular_32_${SIZE_TAG}"
    TACCL_SCHED="$TACCL_DIR/taccl_regular_32_a2a_${SIZE_TAG}.sched"
fi

htsim_build
echo "== regular_32 (degree $DEGREE) A2A $([[ "$NONUNIFORM" == "1" ]] && echo "NONUNIFORM($NONUNI_TAG, pad ${PAD_MB}MB)" || echo "${MESSAGE_SIZE_BYTES}B/pair") packet ${PACKET_SIZE_BYTES}B store_forward=$STORE_FORWARD"

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
            TECCL_WORKFLOW="$NU/teccl_regular_32_a2a_${SIZE_TAG}_workflow.json"
            make_padded_teccl_workflow "$EX/teccl_regular_32_a2a_2mb_workflow.json" "$TECCL_WORKFLOW"
        fi
        teccl_generate "$EX/regular_32_topo.json" "$TECCL_WORKFLOW" "$TECCL_SCHED" >/dev/null
        run_plan_scheduler teccl teccl_schedule "$TECCL_SCHED" ;;
    taccl)
        if [[ "$FORCE_RESOLVE" == "1" || ! -f "$TACCL_SCHED" ]]; then
            mkdir -p "$TACCL_DIR"
            if [[ "$NONUNIFORM" == "1" ]]; then
                TACCL_TOPO="$NU/taccl_regular_32_topo_${SIZE_TAG}.json"
                make_padded_taccl_topo "$EX/taccl_regular_32_topo_2mb.json" "$TACCL_TOPO"
            fi
            (cd "$TACCL_DIR" && "$CONDA" run -n "$CONDA_ENV" taccl solve custom Alltoall \
                --topology-file "$TACCL_TOPO" \
                --sketch-file "$EX/taccl_regular_32_sketch.json" \
                -d "$TACCL_DIR" -f)
            TACCL_JSON="$(ls -t "$TACCL_DIR"/*_taccl.sccl.json | head -1)"
            python3 "$ROOT/tools/taccl_to_htsim_schedule.py" \
                --input "$TACCL_JSON" --output "$TACCL_SCHED" \
                --chunk-size-bytes "$CHUNK_SIZE_BYTES" --topology regular_32
        fi
        run_plan_scheduler taccl taccl_schedule "$TACCL_SCHED" ;;
    *) echo "unknown scheduler: $s" >&2; exit 1 ;;
    esac
done
