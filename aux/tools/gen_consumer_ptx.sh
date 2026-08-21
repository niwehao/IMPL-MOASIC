#!/bin/bash
# 重新生成 src/consumer_kernel_ptx.h(S3 二期消费者内核的嵌入 PTX)。
# 必须用 **CUDA ≤12.6 的 nvcc**(ISA ≤8.5),保证 Raven(cuda/12.6 驱动)
# 与本机都能 JIT;本机 12.8 toolkit 产出 ISA 8.7,Raven 驱动可能拒载。
# 用法:在有合适 nvcc 的机器上(推荐 Raven 登录节点):
#   bash tools/gen_consumer_ptx.sh
set -eu
cd "$(dirname "$0")/../.."
nvcc -ptx -arch=compute_80 -o /tmp/mosaic_consumer_$$.ptx src/consumer_kernel.cu
{
  echo "// 自动生成:tools/gen_consumer_ptx.sh(nvcc 12.6 -ptx -arch=compute_80,"
  echo "// 在 Raven 登录节点生成 —— ISA 8.5,Raven/本机驱动都能 JIT)。"
  echo "// 源 = src/consumer_kernel.cu;改内核先改 .cu 再重新生成本文件。"
  echo "#pragma once"
  echo "static const char* kConsumerPtx ="
  sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/    "/' -e 's/$/\\n"/' /tmp/mosaic_consumer_$$.ptx
  echo "    ;"
} > src/consumer_kernel_ptx.h
rm -f /tmp/mosaic_consumer_$$.ptx
echo "OK: src/consumer_kernel_ptx.h"
