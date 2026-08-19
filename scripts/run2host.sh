#!/usr/bin/env bash
# 双机启动:本机跑 rank 0..G-1,gpu-01 跑 rank G..2G-1。
#   G=4 ./scripts/run2host.sh --sched mosaic --msg 262144
# 坑规避:① 远端命令后的 & 只让远端 bash 返回,本地 ssh 仍会等 fd 关闭,
#          所以本地 ssh 也要放后台;② pkill 用 -x(精确进程名),-f 会匹配到
#          执行 pkill 的那条 bash 自己,把自己杀掉。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$here/hosts.sh"
G=${G:-4}; N=$((2*G)); TMO=${TMO:-90}
JOB=${MOSAIC_JOB:-$RANDOM}
SSH="ssh -o BatchMode=yes $GPU01_SSH"
COMMON="MOSAIC_HOSTS=$GPU02_IP,$GPU01_IP MOSAIC_NRANKS=$N MOSAIC_G=$G MOSAIC_JOB=$JOB MOSAIC_LOG=${MOSAIC_LOG:-1}"
for v in MOSAIC_SLICE_BYTES MOSAIC_CORRIDORS MOSAIC_WAVES MOSAIC_SYNC_US \
         MOSAIC_STAGGER_US MOSAIC_LEDGER MOSAIC_BCAST MOSAIC_STAGE_SLOTS MOSAIC_RELAY_SLOTS \
         MOSAIC_SHM_SLOTS MOSAIC_PORT_BASE \
         MOSAIC_NIC MOSAIC_DETECTOR MOSAIC_PAIR_COLOR MOSAIC_COLD_PACE MOSAIC_SPINEAGG \
         MOSAIC_DROP_A MOSAIC_SLICE_CRC \
         MOSAIC_RDMA_RX_SLOTS MOSAIC_NVSH_SLOTS MOSAIC_WAVE_DEADLINE \
         MOSAIC_ASYNC_SUBMIT MOSAIC_DEMAND_HINT MOSAIC_WAVE_COLLAPSE \
         MOSAIC_CTRL MOSAIC_NVSH_ZCOPY MOSAIC_PHASE \
         MOSAIC_TRACE MOSAIC_TRACE_DIR MOSAIC_TRACE_CAP \
         MOSAIC_DEMAND_SSP MOSAIC_TELEM_TARGETED MOSAIC_CPU_CORE MOSAIC_MEM_BUDGET_MB \
         MOSAIC_ZEROSTREAM MOSAIC_TX_SLOTS MOSAIC_WAVE_FRAG_KB \
         MOSAIC_NET_MODE MOSAIC_LOG \
         NVSHMEM_REMOTE_TRANSPORT NVSHMEM_IB_GID_INDEX NVSHMEM_DISABLE_P2P \
         NVSHMEM_DISABLE_NVLS NVSHMEM_CUMEM_GRANULARITY NVSHMEM_SYMMETRIC_SIZE; do
  [ -n "${!v:-}" ] && COMMON="$COMMON $v=${!v}"
done
cleanup() { $SSH "pkill -9 -x bench_a2a >/dev/null 2>&1; rm -f /dev/shm/mosaic_${JOB}_*" >/dev/null 2>&1; 
            pkill -9 -x bench_a2a >/dev/null 2>&1; rm -f /dev/shm/mosaic_${JOB}_* 2>/dev/null; }
trap cleanup EXIT INT TERM
cleanup; sleep 0.3
# 远端 rank G..2G-1
$SSH "cd $REPO && for ((r=$G;r<$N;r++)); do setsid nohup env $COMMON MOSAIC_RANK=\$r \
      timeout $TMO ./build/bench_a2a $* > /tmp/mosaic_rank\$r.log 2>&1 < /dev/null & done" >/dev/null 2>&1 &
sleep 1.5
# 本机 rank 0..G-1
pids=()
for ((r=0;r<G;r++)); do
  env $COMMON MOSAIC_RANK=$r timeout $TMO ./build/bench_a2a "$@" > /tmp/mosaic_rank$r.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
cat /tmp/mosaic_rank0.log
if [ $rc -ne 0 ] || [ "${VERBOSE:-0}" = 1 ]; then
  echo "----- rc=$rc 本机各 rank 尾部 -----"; for ((r=0;r<G;r++)); do echo "--r$r--"; tail -4 /tmp/mosaic_rank$r.log; done
  echo "----- 远端 -----"; $SSH "for ((r=$G;r<$N;r++)); do echo --r\$r--; tail -4 /tmp/mosaic_rank\$r.log; done" 2>&1
fi
exit $rc
