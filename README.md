# mosaic-rt —— MOSAIC 运行时(EP a2a 通信库)

分布式交换视图 + 运行时动态调整的 alltoallv 运行时:CPU 协议核决定「哪段
字节走哪条路、什么时候发」,GPU 只按描述符搬运。系统总述(构造、Megatron
耦合、模拟器→实机差异、与 NCCL 的差距及原因)见
`../../File/MOSAIC_SYSTEM_0821.md`;实验台账 `../../File/log_raven.md`;
bug/debug 记录 `../../File/log.md`。

## 目录结构

```
mosaic-rt/
├── nccl/           NCCL baseline(bench_nccl.py 同尺墙钟基准 + run2.sh 双机启动;
│                   两后端同一把尺的对比腿,不属于交付物)
├── sim/            MOSAIC 模拟器版本(csg-htsim/FAST/syccl-grb/taccl;
│                   协议核决策语义的金标来源,replay_core_test 对拍它)
├── src/            运行时源码(唯一交付物)
│   ├── api.cc          L3:mosaicAlltoallv[Ex]、完成契约(gated/阻塞)、wait_coll
│   ├── engine.cc       L1+L2:事件循环、波引擎、W 在线自适应、账本、kdesc、
│   │                   走廊承诺位、recv-即-开门、每节点在途帽
│   ├── topo.cc         拓扑/走廊族/账本(NIC 物理单元记账)
│   ├── net.cc          TCP bootstrap、机内 SHM 环/RWIN/DLV、NIC 单元共识
│   ├── verbs.cc        跨机 RDMA 数据面+控制面(host-bounce,IB/RoCE 同码)
│   ├── erpc_ctrl.cc    控制面兼容档(eRPC/DPDK,无 RDMA 卡机器用)
│   ├── consumer_kernel.cu / consumer_kernel_ptx.h
│   │                   GPU 消费者核(唯一搬运工;PTX 由 aux/tools 生成嵌入)
│   └── internal.h      跨文件契约(线格式/Comm/Topo/Ledger/trace 事件)
└── aux/            其余一切(构建产物、工具、测试、绑定)
    ├── build/          构建输出(libmosaic.so、bench_a2a、replay_core_test…)
    ├── include/        对外头文件 mosaic/mosaic.h
    ├── python/         mosaic_py.py(ctypes 绑定)、mosaic_torch/(Megatron
    │                   drop-in:all_to_all_single 同签名 + patch_megatron)
    ├── bench/          bench_a2a.cc(C++ 基准,gated 流序档)
    ├── tests/          replay_core_test(vs 模拟器金标)、test_kcons、
    │                   test_mosaic_torch、fixtures/
    ├── scripts/        本机双机脚本(run2host/local_smoke/sync/sweep)
    ├── tables/         topo_fabric.json 等静态表
    ├── tools/          gen_consumer_ptx.sh(改 .cu 后重新生成 PTX 头)
    └── third_party/    erpc_raven.patch(eRPC 上游补丁;上游源码在 Doc/Code)
```

## 构建

```bash
cd Impl/mosaic-rt && make -j8        # 产物落 aux/build/
# 改了 consumer_kernel.cu:先 bash aux/tools/gen_consumer_ptx.sh(需 nvcc ≤12.6)再 make
```

Raven 上永远不手改代码:本机改 → `bash Impl/common/raven/rpush.sh --build`
(远端重建;rpath 含绝对路径必须目标机编)。

## 使用

**训练接入(Megatron-Core)**——三后端只差一个参数,其余逐字节相同:

```python
import mosaic_torch as mt            # sys.path 加 Impl/mosaic-rt/aux/python
mt.init(backend="mosaic", gpus_per_host=4)   # dist.init_process_group 之后
mt.patch_megatron()                  # 单点替换 mcore 的 a2a 原语;nccl 后端不打
# 环境:MOSAIC_HOSTS=<点分IPv4逗号表> MOSAIC_G=<每机卡数> MOSAIC_JOB=<去撞ID>
```

端到端示例:`Impl/moe-rt/train_moe_mcore.py`(`MT_BACKEND=nccl|mosaic`,
`MT_TOKEN_ZIPF` 构造斜 gating;判据 = 两后端 loss 整列 md5)。

**纯 A2A 基准**(两后端同一把墙钟尺 + 端到端校验):

```bash
# 本机双机:
cd Impl && MT_BACKEND=... bash mosaic-rt/nccl/run2.sh mosaic-rt/nccl/bench_nccl.py --backend mosaic --mode hotdst --alpha 1 --msg 8388608
# C++ 基准(gated 档,引擎内部口径 med_us):
cd Impl/mosaic-rt && G=4 bash aux/scripts/run2host.sh --msg 1048576 --sched mosaic --verify 1
# Raven(占位内):
bash Impl/common/raven/a2a_step.sh <hold_jobid> CELLS="hotdst:8388608:1" BACKENDS="nccl mosaic"
```

**回归**(每次改动后必跑):`cd Impl && bash regression.sh`(20 项:双机
bit-exact、模拟器金标 0 分歧、torch drop-in、训练 loss 一致等)。

**Raven 工作流**:见根 `CLAUDE.md` §2-§3(7 个封装命令;占位 debug 用
`rdev.sh`;进表走独占 sbatch `moe_64gpu.sbatch` / `a2a_16gpu.sbatch`)。

## 旋钮

25 个 env,分四类(平台/容量/决策层/观测),**无一改变数据面走法**;
每个定义处注释「控制什么/默认怎么定/调大调小如何」。清单与三分类见
`../../File/MOSAIC_SYSTEM_0821.md` §5;grep 审计:
`grep -rn getenv src/ | grep -v "TRACE\|PHASE\|LOG\|DUMP"`。

## 完成契约与铁律(读源码前必知)

- 给流 = gated 流序档(提交即返回,门核等 done_flag,**recv 就绪即开门**);
  不给流 = host 阻塞档。一代在飞:下一次提交先收尾上一笔。
- **门未决窗口内引擎零 CUDA 调用**(G.1:与 app 线程 cudaFree 驱动锁互锁,
  两次事故实录);消费者核每 coll 起停绝不常驻(坑 #36)。
- 决策只用本地观测 + 显式交换的状态,无零成本全局同步假设(G.2)。
