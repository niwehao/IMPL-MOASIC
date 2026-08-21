#!/usr/bin/env bash
# 单机 n rank 冒烟。每个 rank 自带 timeout,退出时清干净 —— 否则被 timeout 杀掉的
# 是脚本、子进程会变孤儿继续占着 31000+ 的固定端口,下一次运行就会连到"上一次的
# rank 0"上,表现为诡异的建链挂死(nano-vllm 报告的 G12 同款坑)。
set -uo pipefail
N=${N:-4}; G=${G:-4}; TMO=${TMO:-60}
export MOSAIC_HOSTS=${MOSAIC_HOSTS:-127.0.0.1} MOSAIC_NRANKS=$N MOSAIC_G=$G
export MOSAIC_JOB=${MOSAIC_JOB:-$$} MOSAIC_LOG=${MOSAIC_LOG:-1}
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill -9 $p 2>/dev/null; done; rm -f /dev/shm/mosaic_${MOSAIC_JOB}_* 2>/dev/null; }
trap cleanup EXIT INT TERM
for ((r=0;r<N;r++)); do
  MOSAIC_RANK=$r timeout $TMO ./aux/build/bench_a2a "$@" > /tmp/mosaic_r$r.log 2>&1 & pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=$?; done
cat /tmp/mosaic_r0.log
[ $rc -ne 0 ] && { echo "=== rc=$rc,各 rank 日志尾部 ==="; for ((r=0;r<N;r++)); do echo "--- r$r ---"; tail -6 /tmp/mosaic_r$r.log; done; }
exit $rc
