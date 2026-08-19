# IMPL-MOASIC

MOSAIC 通信运行时(`libmosaic`)的实现与测试代码 —— 一个面向 MoE alltoallv 的
**分布式交换视图 + 运行时动态调整**的集合通信库。

## 分层

`src/internal.h` 抬头把分层写死成四层,**平台差异只允许出现在 L0**:

| 层 | 内容 | 文件 |
|---|---|---|
| **L3** API | C ABI,`mosaicAlltoallv` 与 `MPI_Alltoallv` 同语义 | `include/mosaic/mosaic.h`,`src/api.cc` |
| **L2** 协议核 | **纯 CPU,不碰 CUDA**:Topo/走廊族 · Ledger 注水 · DstCtrl 波引擎 · 通道 A(遥测)/ B(策略)· 错峰 · 检测器 · PC/PACE 门 | `src/topo.cc`,`src/engine.cc` |
| **L1** 执行引擎 | 单 event 线程:slice 状态机 + CUDA 异步拷贝提交 + token bucket | `src/engine.cc`,`src/consumer_kernel.cu` |
| **L0** 传输 | SHM/CUDA-IPC(机内) \| RoCE verbs \| NVSHMEM \| TCP(兜底) | `src/net.cc`,`src/verbs.cc`,`src/nvshmem_pipe.cc`,`src/erpc_ctrl.cc` |

**线程模型**:app 线程(调用者)+ 1 个 event 线程。协议决策、CUDA 提交、socket
收发全在 event 线程里做 —— 单线程事件循环 = 无锁 = 决策序列可复现,这是能离线
回放、能与模拟器对拍的前提。

调度器可换头(`--sched`),但**发射路径只有一份**(`emit_range_pub` /
`emit_chain_pub`):换调度器只换 L2 的决策,L1/L0 逐字节相同。

## 目录

```
include/mosaic/   公开 C ABI
src/              运行时本体(L0-L3)
python/           ctypes 绑定 + Megatron-Core a2a 原语的单点替换
bench/            bench_a2a —— 各调度方案共用的 alltoallv 基准 harness
tests/            test_topo(L2 离线单测)· replay_core_test(离线回放)
                  test_kcons · test_persist_sync · test_devpipe · pytest
tools/            hw_bounds.py(割下界)· gen_consumer_ptx.sh
scripts/          本机单机 / 双机跑测脚本
```

## 构建

```bash
make -j8              # 产出 build/libmosaic.so 与各测试二进制
```

外部依赖(不在本仓库内):

- **CUDA** —— `CUDA_ROOT`,默认 `$HOME/cuda-12.8`。只链 `libcudart`,不需要 nvcc。
- **libibverbs** —— RoCE 数据面。
- **FAST 官方源码** —— `../fast-rt/FAST/amd/`。`fast` 调度基线**直接编译未拷贝、
  未修改的官方源文件**(公平性要求),因此本仓库不 vendor 它;缺失时该目标编不过。
  `src/fast_native/hip/hip_runtime.h` 是为此提供的空编译 shim(不是 HIP)。
- **NVSHMEM**(可选)—— `MOSAIC_NIC=nvshmem` 数据面。
- **eRPC**(可选)—— `make ERPC=1`,`MOSAIC_CTRL=erpc` 控制面。

## 跑测

```bash
./build/test_topo                                    # L2 走廊族离线单测(无 GPU / 无网络)
./build/replay_core_test tests/fixtures/demand_n8_zipf1.0_seed42.txt \
                         tests/fixtures/trace_n8_zipf1.0_seed42_nv400nic100taper180.txt 400 100 180
N=4 G=4 ./scripts/local_smoke.sh --msg 1048576 --sched mosaic --verify 1    # 单机 4 rank
G=4 ./scripts/run2host.sh --msg 1048576 --sched mosaic --verify 1           # 双机 8 rank
```

`scripts/hosts.sh` 是双机拓扑的唯一事实源(内网地址,按自己的环境改)。

## 旋钮

全部配置字段都有 `MOSAIC_*` 环境变量镜像,默认值见
`mosaicConfigDefault()`(`src/util.cc`)。常用的几个:

| 旋钮 | 默认 | 作用 |
|---|---|---|
| `MOSAIC_SCHED` | mosaic | 调度器:`mosaic` / `direct` / `fast` |
| `MOSAIC_NIC` | verbs | 数据面:`verbs` / `nvshmem` / `tcp` |
| `MOSAIC_CTRL` | tcp | 控制面:`tcp` / `verbs` / `erpc` |
| `MOSAIC_SLICE_BYTES` | 1 MiB | 传输层分片大小 |
| `MOSAIC_THROTTLE` | 1 | `0` = 额定值只喂决策,不做限速 |
| `MOSAIC_PHASE` | 0 | `1` = 每 collective 打一行相位分解 |
| `MOSAIC_TRACE` | 0 | `1` = 32B 定长事件 mmap 环(崩溃也留最近窗口) |
