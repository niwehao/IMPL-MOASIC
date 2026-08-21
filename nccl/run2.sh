#!/usr/bin/env bash
# world=2 双机跑一条 python 命令(rank0 本机 / rank1 gpu-01)。
# 沿用 nano-vllm 的 env 与"本地 ssh 也要放后台"的技巧。
set -uo pipefail
REPO=$HOME/CodeRead/xccl/Impl
V=$HOME/CodeRead/nano-vllm/.venv/bin/python
SSH="ssh -o BatchMode=yes 192.168.100.1"
export MASTER_ADDR=192.168.100.2 MASTER_PORT=${MASTER_PORT:-29511} WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=ens5f0np0 GLOO_SOCKET_IFNAME=ens5f0np0
export NCCL_IB_GID_INDEX=3 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=SYS
export MOSAIC_HOSTS=192.168.100.2,192.168.100.1 MOSAIC_G=1 MOSAIC_JOB=${MOSAIC_JOB:-7}
FWD="MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=2 \
 NCCL_SOCKET_IFNAME=ens5f0np0 GLOO_SOCKET_IFNAME=ens5f0np0 NCCL_IB_GID_INDEX=3 \
 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=SYS NCCL_IB_HCA=rocep66s0f0 \
 MOSAIC_HOSTS=$MOSAIC_HOSTS MOSAIC_G=1 MOSAIC_JOB=$MOSAIC_JOB ${EXTRA_ENV:-}"
export NCCL_IB_HCA=mlx5_0     # 两机 verbs 名不同,必须分支
# 本地**不能**用 pkill -f:调用本脚本的那条命令行里就含脚本参数(如 bench_nccl.py),
# 会把调用者自己杀掉(踩过两次)。本地按 PID 杀,远端才用 pkill(远端那条 bash 的
# 命令行是带中括号的 pattern 本身,匹配不上自己)。
LOCAL_PID=""
# 清理必须覆盖**所有**经本脚本跑过的 python 入口:上一轮挂死的进程会占住
# mosaic 的固定端口(31000+rank),下一轮连上"上一次的 rank"表现为诡异的
# 协议错乱(posted 计数翻倍、recv 恒 0)—— 踩过,mosaic-rt/log.md 坑 #19。
# 本地 pattern 必须带 python 前缀:run2.sh 自己的命令行里就含测试文件名,
# 裸 pattern 会把自己杀掉(脚本头部警告过的同一坑)。
KILLPAT='bench[_]nccl|test[_]mosaic|bench[_]moe|dbg[_]fast|train[_]moe'
cleanup(){ $SSH "pkill -9 -f '$KILLPAT' >/dev/null 2>&1" >/dev/null 2>&1
           pkill -9 -f "bin/python .*($KILLPAT)" 2>/dev/null
           [ -n "$LOCAL_PID" ] && kill -9 $LOCAL_PID 2>/dev/null; return 0; }
trap cleanup EXIT INT TERM
cleanup; sleep 0.3
$SSH "cd $REPO && setsid nohup env $FWD RANK=1 $V $* > /tmp/bl_rank1.log 2>&1 < /dev/null &" >/dev/null 2>&1 &
sleep 3
RANK=0 timeout ${TMO:-300} $V "$@" & LOCAL_PID=$!
wait $LOCAL_PID
rc=$?
[ "${VERBOSE:-0}" = 1 ] && { echo "--- rank1 ---"; $SSH "tail -5 /tmp/bl_rank1.log"; }
exit $rc
