#!/usr/bin/env bash
# gpu-02 → gpu-01 单向同步**整个 Impl 树**。约定:gpu-01 上不手工改任何东西。
#
# /home 不是共享盘(实测),所以每次改代码都必须同步 —— 这是本项目的头号操作坑:
# 忘了同步的表现是"对端跑的是旧二进制",而症状会伪装成协议 bug。
#
# 为什么同步整棵 Impl 而不只是 mosaic-rt:moe-rt / baseline-rt / fast-rt 的
# worker 都在远端起,四个目录必须版本一致。早先只同步 mosaic-rt、其余手工 rsync,
# 结果 python 绑定改了却没过去,报成了"远端 import 失败"。
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$here/hosts.sh"
SSH() { ssh -o BatchMode=yes "$@"; }

SSH $GPU01_SSH "mkdir -p $IMPL"
# 排除构建产物:.so/.o 会被远端自己重建(rpath 里有绝对路径,必须本地重建)。
# 例外 fast-rt/FAST/nvidia/libflash.so —— 那个远端不重建,直接用同步过去的。
rsync -a --info=stats1 -e "ssh -o BatchMode=yes" --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'mosaic-rt/aux/build' --exclude 'mosaic-rt/src/*.o' --exclude 'mosaic-rt/src/*.d' \
  --exclude 'mosaic-rt/results' --exclude 'fast-rt/FAST/nvidia/build' \
  "$IMPL/" "$GPU01_SSH:$IMPL/"

# 远端重建 mosaic-rt(rpath 含绝对路径,必须在目标机上生成)。
# 失败必须当场炸出来:管道 `| tail` 会吞掉 make 的退出码,远端就会静默留在
# 上一次构建的旧二进制上,而症状会伪装成协议 bug(和忘同步是同一类坑)。
if ! SSH $GPU01_SSH "cd $REPO && make -j8 > /tmp/mosaic_remote_make.log 2>&1"; then
  echo "!! 远端 make 失败,尾部日志:" >&2
  SSH $GPU01_SSH "tail -30 /tmp/mosaic_remote_make.log" >&2
  exit 1
fi
SSH $GPU01_SSH "tail -2 /tmp/mosaic_remote_make.log"
