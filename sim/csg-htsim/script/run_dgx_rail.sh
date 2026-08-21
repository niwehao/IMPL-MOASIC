#!/usr/bin/env bash
# dgx_rail 五方案 A2A 对比:编译 -> schedule 生成(带缓存)-> htsim 模拟。
# 拓扑:4 主机 x 8 GPU rail-optimized,8 轨 leaf + 单 spine taper。
# NONUNIFORM=1 切换到非均匀 A2A(hot-dst zipf):ecmp/fast 直接吃 demand 矩阵;
# teccl/taccl/syccl 先在 pow2 padded 尺寸上求解均匀计划,再用 fill 工具截断
# 成实际 demand(计划只当"填充容量",src 只发实际字节)。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ===================== 可调参数 =====================
NODES="${NODES:-32}"
TOPOLOGY="${TOPOLOGY:-dgx_rail}"
PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES:-1024}"
MESSAGE_SIZE_BYTES="${MESSAGE_SIZE_BYTES:-2097152}"   # 每个 P2P 对的字节数
CHUNK_SIZE_BYTES="${CHUNK_SIZE_BYTES:-2097152}"
STORE_FORWARD="${STORE_FORWARD:-1}"
DGX_TAPER_GBPS="${DGX_TAPER_GBPS:-180}"               # leaf->spine 收敛(4:1 @180)
SCHEDULERS="${SCHEDULERS:-static_ecmp teccl taccl syccl fast}"
FORCE_RESOLVE="${FORCE_RESOLVE:-0}"                   # 1 = 忽略缓存重新求解
NONUNIFORM="${NONUNIFORM:-0}"                         # 1 = 非均匀 A2A
# TACCL/SyCCL 求解时限
export TACCL_SCHEDULER_TIME_LIMIT="${TACCL_SCHEDULER_TIME_LIMIT:-600}"
export TACCL_ROUTING_TIME_LIMIT="${TACCL_ROUTING_TIME_LIMIT:-600}"
# ====================================================

EXTRA_TOPO_ARGS="-dgx_taper_gbps $DGX_TAPER_GBPS"
EX="$ROOT/examples/dgx_rail"
NU="$ROOT/cache/nonuniform"

if [[ "$NONUNIFORM" == "1" ]]; then
    nonuniform_setup
    SIZE_TAG="${PAD_MB}mb"
else
    DEMAND_FILE=""
    SIZE_TAG="2mb"
fi

TECCL_WORKFLOW="$EX/teccl_dgx_rail_a2a_2mb_workflow.json"
TECCL_SCHED="$EX/teccl_dgx_rail_a2a_${SIZE_TAG}.sched"
TACCL_TOPO="$EX/taccl_dgx_rail_topo_2mb.json"
TACCL_DIR="$ROOT/cache/taccl_dgx_rail"
TACCL_SCHED="$TACCL_DIR/taccl_dgx_rail_a2a_${SIZE_TAG}.sched"
SYCCL_CONFIG="$ROOT/../syccl-grb/config/rail-4h8g-a2a-2mb.json"
SYCCL_RESULT="$ROOT/../syccl-grb/build/result-rail-4h8g-a2a-${SIZE_TAG}.json"
SYCCL_SCHED="$ROOT/cache/syccl_dgx_rail_a2a_${SIZE_TAG}.sched"
if [[ "$NONUNIFORM" == "1" ]]; then
    TECCL_SCHED="$NU/teccl_dgx_rail_a2a_${SIZE_TAG}.sched"
    TACCL_DIR="$NU/taccl_dgx_rail_${SIZE_TAG}"
    TACCL_SCHED="$TACCL_DIR/taccl_dgx_rail_a2a_${SIZE_TAG}.sched"
    SYCCL_SCHED="$NU/syccl_dgx_rail_a2a_${SIZE_TAG}.sched"
fi

htsim_build
echo "== dgx_rail A2A $([[ "$NONUNIFORM" == "1" ]] && echo "NONUNIFORM($NONUNI_TAG, pad ${PAD_MB}MB)" || echo "${MESSAGE_SIZE_BYTES}B/pair") packet ${PACKET_SIZE_BYTES}B store_forward=$STORE_FORWARD taper=${DGX_TAPER_GBPS}G"

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
            TECCL_WORKFLOW="$NU/teccl_dgx_rail_a2a_${SIZE_TAG}_workflow.json"
            make_padded_teccl_workflow "$EX/teccl_dgx_rail_a2a_2mb_workflow.json" "$TECCL_WORKFLOW"
        fi
        # 生成器自带内容寻址缓存,输入不变即缓存命中
        teccl_generate "$EX/dgx_rail_topo.json" "$TECCL_WORKFLOW" "$TECCL_SCHED" >/dev/null
        run_plan_scheduler teccl teccl_schedule "$TECCL_SCHED" ;;
    taccl)
        if [[ "$FORCE_RESOLVE" == "1" || ! -f "$TACCL_SCHED" ]]; then
            mkdir -p "$TACCL_DIR"
            if [[ "$NONUNIFORM" == "1" ]]; then
                TACCL_TOPO="$NU/taccl_dgx_rail_topo_${SIZE_TAG}.json"
                make_padded_taccl_topo "$EX/taccl_dgx_rail_topo_2mb.json" "$TACCL_TOPO"
            fi
            (cd "$TACCL_DIR" && "$CONDA" run -n "$CONDA_ENV" taccl solve custom Alltoall \
                --topology-file "$TACCL_TOPO" \
                --sketch-file "$EX/taccl_dgx_rail_sketch.json" \
                -d "$TACCL_DIR" -f)
            TACCL_JSON="$(ls -t "$TACCL_DIR"/*_taccl.sccl.json | head -1)"
            python3 "$ROOT/tools/taccl_to_htsim_schedule.py" \
                --input "$TACCL_JSON" --output "$TACCL_SCHED" \
                --chunk-size-bytes "$CHUNK_SIZE_BYTES" --topology dgx_rail
        fi
        run_plan_scheduler taccl taccl_schedule "$TACCL_SCHED" ;;
    syccl)
        if [[ "$FORCE_RESOLVE" == "1" || ! -f "$SYCCL_SCHED" ]]; then
            if [[ "$FORCE_RESOLVE" == "1" || ! -f "$SYCCL_RESULT" ]]; then
                if [[ "$NONUNIFORM" == "1" ]]; then
                    SYCCL_CONFIG="$ROOT/../syccl-grb/config/rail-4h8g-a2a-${SIZE_TAG}.json"
                    make_padded_syccl_config \
                        "$ROOT/../syccl-grb/config/rail-4h8g-a2a-2mb.json" \
                        "$SYCCL_CONFIG" "result-rail-4h8g-a2a-${SIZE_TAG}.json"
                fi
                (cd "$ROOT/../syccl-grb/build" \
                    && ./synthesize -f "$SYCCL_CONFIG" solve)
            fi
            python3 "$ROOT/tools/syccl_generate_schedule.py" \
                --syccl-result "$SYCCL_RESULT" \
                --explicit-path-topology dgx_rail --output "$SYCCL_SCHED"
        fi
        run_plan_scheduler syccl syccl_schedule "$SYCCL_SCHED" ;;
    *) echo "unknown scheduler: $s" >&2; exit 1 ;;
    esac
done
