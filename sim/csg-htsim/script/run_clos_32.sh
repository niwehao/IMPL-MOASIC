#!/usr/bin/env bash
# clos_32 五方案 A2A 对比:编译 -> schedule 生成(带缓存)-> htsim 模拟。
# 拓扑:纯两层 clos(switch-as-wire):32 host(单条 100G 上行,无机内互连),
# 4 leaf x 8 host,单 spine,leaf 上行 400G(2:1 收敛)。
# 注意:host 度数=1,对称 A2A 下所有方案同受 host/leaf 上行 cut 约束,
# 这是"公共集合可执行性"拓扑,不用于展示 solver 优势
# (见 Note/solver_topology_capabilities.md)。FAST 在此退化为纯 Birkhoff
# 置换调度(gpus_per_server=1,无 scale-up 层可用)。
# NONUNIFORM=1:非均匀 A2A;teccl/taccl/syccl 用 pow2 padded 计划 + fill 截断。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ===================== 可调参数 =====================
NODES="${NODES:-32}"
TOPOLOGY="${TOPOLOGY:-clos}"
PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES:-1024}"
MESSAGE_SIZE_BYTES="${MESSAGE_SIZE_BYTES:-2097152}"
CHUNK_SIZE_BYTES="${CHUNK_SIZE_BYTES:-2097152}"
STORE_FORWARD="${STORE_FORWARD:-1}"
CLOS_LEAFS="${CLOS_LEAFS:-4}"
CLOS_SPINES="${CLOS_SPINES:-1}"
CLOS_UPLINK_GBPS="${CLOS_UPLINK_GBPS:-400}"       # 0 = auto (2:1)
SCHEDULERS="${SCHEDULERS:-static_ecmp teccl taccl syccl fast}"
FORCE_RESOLVE="${FORCE_RESOLVE:-0}"
NONUNIFORM="${NONUNIFORM:-0}"
export TACCL_SCHEDULER_TIME_LIMIT="${TACCL_SCHEDULER_TIME_LIMIT:-600}"
export TACCL_ROUTING_TIME_LIMIT="${TACCL_ROUTING_TIME_LIMIT:-600}"
# ====================================================

EXTRA_TOPO_ARGS="-clos_leafs $CLOS_LEAFS -clos_spines $CLOS_SPINES -clos_uplink_gbps $CLOS_UPLINK_GBPS"
EX="$ROOT/examples/clos_32"
NU="$ROOT/cache/nonuniform"

if [[ "$NONUNIFORM" == "1" ]]; then
    nonuniform_setup
    SIZE_TAG="${PAD_MB}mb"
else
    DEMAND_FILE=""
    SIZE_TAG="2mb"
fi

TECCL_WORKFLOW="$EX/teccl_clos_32_a2a_2mb_workflow.json"
TECCL_SCHED="$EX/teccl_clos_32_a2a_${SIZE_TAG}.sched"
TACCL_TOPO="$EX/taccl_clos_32_topo_2mb.json"
TACCL_DIR="$ROOT/cache/taccl_clos_32"
TACCL_SCHED="$TACCL_DIR/taccl_clos_32_a2a_${SIZE_TAG}.sched"
SYCCL_CONFIG="$ROOT/../syccl-grb/config/clos-32-a2a-2mb.json"
SYCCL_RESULT="$ROOT/../syccl-grb/build/result-clos-32-a2a-${SIZE_TAG}.json"
SYCCL_SCHED="$ROOT/cache/syccl_clos_32_a2a_${SIZE_TAG}.sched"
if [[ "$NONUNIFORM" == "1" ]]; then
    TECCL_SCHED="$NU/teccl_clos_32_a2a_${SIZE_TAG}.sched"
    TACCL_DIR="$NU/taccl_clos_32_${SIZE_TAG}"
    TACCL_SCHED="$TACCL_DIR/taccl_clos_32_a2a_${SIZE_TAG}.sched"
    SYCCL_SCHED="$NU/syccl_clos_32_a2a_${SIZE_TAG}.sched"
fi

htsim_build
echo "== clos_32 A2A $([[ "$NONUNIFORM" == "1" ]] && echo "NONUNIFORM($NONUNI_TAG, pad ${PAD_MB}MB)" || echo "${MESSAGE_SIZE_BYTES}B/pair") packet ${PACKET_SIZE_BYTES}B store_forward=$STORE_FORWARD leafs=$CLOS_LEAFS spines=$CLOS_SPINES uplink=${CLOS_UPLINK_GBPS}G"

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
            TECCL_WORKFLOW="$NU/teccl_clos_32_a2a_${SIZE_TAG}_workflow.json"
            make_padded_teccl_workflow "$EX/teccl_clos_32_a2a_2mb_workflow.json" "$TECCL_WORKFLOW"
        fi
        teccl_generate "$EX/clos_32_topo.json" "$TECCL_WORKFLOW" "$TECCL_SCHED" >/dev/null
        run_plan_scheduler teccl teccl_schedule "$TECCL_SCHED" ;;
    taccl)
        if [[ "$FORCE_RESOLVE" == "1" || ! -f "$TACCL_SCHED" ]]; then
            mkdir -p "$TACCL_DIR"
            if [[ "$NONUNIFORM" == "1" ]]; then
                TACCL_TOPO="$NU/taccl_clos_32_topo_${SIZE_TAG}.json"
                make_padded_taccl_topo "$EX/taccl_clos_32_topo_2mb.json" "$TACCL_TOPO"
            fi
            (cd "$TACCL_DIR" && "$CONDA" run -n "$CONDA_ENV" taccl solve custom Alltoall \
                --topology-file "$TACCL_TOPO" \
                --sketch-file "$EX/taccl_clos_32_sketch.json" \
                -d "$TACCL_DIR" -f)
            TACCL_JSON="$(ls -t "$TACCL_DIR"/*_taccl.sccl.json | head -1)"
            python3 "$ROOT/tools/taccl_to_htsim_schedule.py" \
                --input "$TACCL_JSON" --output "$TACCL_SCHED" \
                --chunk-size-bytes "$CHUNK_SIZE_BYTES" --topology clos_32
        fi
        run_plan_scheduler taccl taccl_schedule "$TACCL_SCHED" ;;
    syccl)
        if [[ "$FORCE_RESOLVE" == "1" || ! -f "$SYCCL_SCHED" ]]; then
            if [[ "$FORCE_RESOLVE" == "1" || ! -f "$SYCCL_RESULT" ]]; then
                if [[ "$NONUNIFORM" == "1" ]]; then
                    SYCCL_CONFIG="$ROOT/../syccl-grb/config/clos-32-a2a-${SIZE_TAG}.json"
                    make_padded_syccl_config \
                        "$ROOT/../syccl-grb/config/clos-32-a2a-2mb.json" \
                        "$SYCCL_CONFIG" "result-clos-32-a2a-${SIZE_TAG}.json"
                fi
                (cd "$ROOT/../syccl-grb/build" \
                    && ./synthesize -f "$SYCCL_CONFIG" solve)
            fi
            python3 "$ROOT/tools/syccl_generate_schedule.py" \
                --syccl-result "$SYCCL_RESULT" \
                --explicit-path-topology clos_32 --output "$SYCCL_SCHED"
        fi
        run_plan_scheduler syccl syccl_schedule "$SYCCL_SCHED" ;;
    *) echo "unknown scheduler: $s" >&2; exit 1 ;;
    esac
done
