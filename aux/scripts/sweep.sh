#!/usr/bin/env bash
# NV:NIC 扫描 × 三方案(Imp_Plan/07 §2.1 的主实验)。
#
# 模拟包线(由 scripts/calibrate.sh 实测确定,见 log.md「标定」):
#   物理上限:SHM/PCIe ≈ 28 Gbps/rank(4 rank 聚合 14 GB/s);TCP ≈ 9.2 Gbps/rank。
#   忠实性要求**限速器必须是绑定约束**,所以取 vNV ≤ 16、vNIC ≤ 2.5,留 2-4× 余量。
#   —— 额定值一旦贴近物理上限,测的就不是调度算法而是 PCIe/TCP 的实现开销了。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here/.."
OUT=${OUT:-results/sweep_$(date +%m%d_%H%M).csv}
mkdir -p "$(dirname "$OUT")"
MSG=${MSG:-1048576}; ITERS=${ITERS:-5}; WARM=${WARM:-2}
MODE=${MODE:-hotdst}; ALPHA=${ALPHA:-1.0}; SEED=${SEED:-42}
# **2026-08-17 用户指令:进表只比三套完整方案** —— MOSAIC(我们的调度器+我们的后端)/
# NCCL(官方)/ FAST(**官方 libflash 全栈**,不是"FAST 调度器+我们的后端")。
# 因此默认 SCHEDS 只剩 mosaic:`direct` 已删除;bench_a2a 里的 `fast` 是那个被删的
# 混合身份(官方调度器接我们的数据面),**不再作为对比项**,只保留为消融/诊断用。
SCHEDS=${SCHEDS:-"mosaic"}
# "比值:vNV:vNIC"
POINTS=${POINTS:-"13.3:16:1.2 8:16:2.0 4:8:2.0 2:4:2.0 1:2:2.0 0.5:1:2.0"}

hdr=0
for pt in $POINTS; do
  IFS=: read -r ratio nv nic <<< "$pt"
  for s in $SCHEDS; do
    line=$(G=4 TMO=${TMO:-120} MOSAIC_LOG=1 timeout $((${TMO:-120}+30)) \
      ./scripts/run2host.sh --msg $MSG --iters $ITERS --warmup $WARM --sched $s \
      --nv $nv --nic $nic --mode $MODE --alpha $ALPHA --seed $SEED --tag "r${ratio}" 2>&1 | grep -E "^tag,|^r${ratio},")
    if [ -z "$line" ]; then echo "  [!] ratio=$ratio $s FAILED" >&2; continue; fi
    if [ $hdr -eq 0 ]; then echo "ratio,$(echo "$line"|head -1)" > "$OUT"; hdr=1; fi
    echo "$ratio,$(echo "$line"|tail -1)" >> "$OUT"
    echo "  ratio=$ratio $s: $(echo "$line"|tail -1|cut -d, -f13)us"
  done
done
echo "→ $OUT"
column -s, -t < "$OUT" | cut -c1-190
