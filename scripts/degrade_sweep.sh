#!/usr/bin/env bash
# 非通告 NIC 降速实验(Imp_Plan/07 §2.1 注入轴;报告 §11.4 降速族的真机版)。
# 语义:MOSAIC_DEGRADE_RANK 只降物理 token bucket,额定/账本/下界仍用通告值,
# 调度层只能靠 NIC 域检测器(0.45×/3 折/quorum≤2)自己发现 —— "非通告"。
#   hot  = demand 列和最大的 dst(hotdst α=1 seed42 → rank 0)
#   cold = 列和最小的 dst(→ rank 6)
# 方案:direct / fast(官方调度器) / mosaic / mosaic-det0(检测器消融,
#   验证"遥测是检测的前提")。每 cell 预热 1 + 正式 ${ITERS:-5} 取 median。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-$here/../results/degrade.csv}
ITERS=${ITERS:-5}
HOT=${HOT:-0}; COLD=${COLD:-6}
: > "$OUT.tmp"
run_cell() {  # $1=cell  $2=结果里的方案名  $3=真实 --sched  其余 = 额外 env
  local cell=$1 label=$2 sched=$3; shift 3
  local line
  line=$(env "$@" G=4 TMO=300 bash "$here/run2host.sh" --sched "$sched" --mode hotdst \
          --msg 2097152 --nv 8 --nic 2 --iters $ITERS --warmup 1 --verify 1 \
          --tag "$cell" 2>/dev/null | grep -m1 -E "^$cell,")
  if [ -z "$line" ]; then
    echo "$cell,$label,FAILED" | tee -a "$OUT.tmp"
    return 1
  fi
  # 把方案列(第 2 列)替换为 label(mosaic-det0 与 mosaic 的区分)
  echo "$line" | awk -F, -v l="$label" 'BEGIN{OFS=","} {$2=l; print}' | tee -a "$OUT.tmp"
}
for CELL in none hot4 hot2 cold4 cold2; do
  case $CELL in
    none)  ENV=(_=1) ;;
    hot4)  ENV=(MOSAIC_DEGRADE_RANK=$HOT MOSAIC_DEGRADE_FACTOR=4) ;;
    hot2)  ENV=(MOSAIC_DEGRADE_RANK=$HOT MOSAIC_DEGRADE_FACTOR=2) ;;
    cold4) ENV=(MOSAIC_DEGRADE_RANK=$COLD MOSAIC_DEGRADE_FACTOR=4) ;;
    cold2) ENV=(MOSAIC_DEGRADE_RANK=$COLD MOSAIC_DEGRADE_FACTOR=2) ;;
  esac
  run_cell "$CELL" direct      direct "${ENV[@]}"
  run_cell "$CELL" fast        fast   "${ENV[@]}"
  run_cell "$CELL" mosaic      mosaic "${ENV[@]}"
  run_cell "$CELL" mosaic-det0 mosaic "${ENV[@]}" MOSAIC_DETECTOR=0
done
mv "$OUT.tmp" "$OUT"
echo "== 结果写入 $OUT =="
