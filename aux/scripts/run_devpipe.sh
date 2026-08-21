#!/usr/bin/env bash
# run_devpipe.sh —— 双机跑 test_devpipe(多 wave 数据面 A/B/C 微基准)。
# 沿用 run2host.sh / run_fast_smoke.sh 的双机与清场惯例(坑 #19:
# 残留进程会占住端口毒化下一轮;本地按 PID 杀,远端才用 pkill)。
#   用法: ./scripts/run_devpipe.sh [--K 32 --W 16 --sizes a,b,c --modes host,onstream,dev]
#   环境: TRANSPORT=ibrc|ibgda(默认 ibrc;ibgda = 设备直接敲门铃,真零主机)
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SSH="ssh -o BatchMode=yes 192.168.100.1"
PORT=${PORT:-29650}
TRANSPORT=${TRANSPORT:-ibrc}
# 两机 verbs 设备名不同,必须分支(run2.sh 同款事实)
HCA_LOCAL=${HCA_LOCAL:-mlx5_0}
HCA_REMOTE=${HCA_REMOTE:-rocep66s0f0}
E="NVSHMEM_REMOTE_TRANSPORT=$TRANSPORT NVSHMEM_IB_GID_INDEX=3 NVSHMEM_DISABLE_P2P=1 \
 NVSHMEM_QP_DEPTH=2048 NVSHMEM_DISABLE_NVLS=1 NVSHMEM_CUMEM_GRANULARITY=536870912 \
 NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-2147483648}"
[ "$TRANSPORT" = "ibgda" ] && E="$E NVSHMEM_IB_ENABLE_IBGDA=1"

LOCAL_PID=""
cleanup(){ $SSH "pkill -9 -x test[_]devpipe >/dev/null 2>&1" >/dev/null 2>&1
           [ -n "$LOCAL_PID" ] && kill -9 "$LOCAL_PID" 2>/dev/null
           pkill -9 -x test_devpipe 2>/dev/null; return 0; }
trap cleanup EXIT INT TERM
cleanup; sleep 0.3

$SSH "cd $REPO && setsid nohup env $E NVSHMEM_HCA_LIST=$HCA_REMOTE:1 \
      ./aux/build/test_devpipe --rank 1 --peer 192.168.100.2 --port $PORT $* \
      > /tmp/devpipe_r1.log 2>&1 < /dev/null &" >/dev/null 2>&1 &
sleep 2
cd "$REPO" && env $E NVSHMEM_HCA_LIST=$HCA_LOCAL:1 \
  timeout "${TMO:-300}" ./aux/build/test_devpipe --rank 0 --peer 192.168.100.2 --port $PORT "$@" &
LOCAL_PID=$!
wait $LOCAL_PID; rc=$?
[ "${VERBOSE:-0}" = 1 ] && { echo "--- rank1 ---"; $SSH "tail -8 /tmp/devpipe_r1.log"; }
exit $rc
