# 当前方案的实验设置与结果(2026-08-25)

> 本文只描述**当前交付版本**(接收端信用反压 + W 在线自适应 +
> recv-即-开门 + src 驱动 dispatch + 统一搬运 API,commit 8d6187a)的
> 设置与数字。历史版本与废案见 note/timeline.md。所有数字带实验编号,
> 原始数据在 `Result/<编号>/`。

## 1. 实验平台

| 项 | 值 |
|---|---|
| 集群 | MPCDF Raven,GPU 分区 `gpu` 档 |
| 形态(铁律) | **4 节点 × 4 GPU = 16 rank**,同档硬件,形态核验(IB 口数)不过不出数 |
| GPU | A100-SXM4 40GB × 4/节点,NVLink 全互连(sm_80) |
| 网络 | **每节点 1 × InfiniBand HDR200(4 卡共享)**,2 级胖树(12 leaf → 1 spine);实测 host↔host 线速 197.3 Gbps |
| CPU/内存 | 72 核 / 512 GB 每节点 |
| 软件 | CUDA 12.6 / gcc 13 / torch 2.8.0+cu128 / NCCL 2.27.3 / Megatron-Core 0.13.1 |
| 对比对象 | 原生 NCCL(原生控制面 + 原生数据面,零改动) |

**进表纪律**:凡进对比表的数字,一律出自**独占 sbatch 作业**(非占位
`--overlap` 步),4×4 形态核验通过,且通过端到端数据校验。

## 2. 负载与口径

### 2.1 纯 A2A 基准(`nccl/bench_nccl.py`,两后端同一把尺)

- **负载**:`build_demand(n=16, msg, alpha, seed=42)` 生成 demand 矩阵;
  `uniform` = 每对等量;`hotdst α` = 目的列按 Zipf(α)倾斜
  (α=1 中度 incast,热 rank 收 587MB;α=2 极端 incast,热 rank 收 1.22GB)。
- **计时**:barrier 对齐后逐 iter 墙钟,120 iter 取中位;
  warmup 5 不计。两后端同一进程框架、同一矩阵、同一秒表。
- **两个口径**(表里分开写):
  - `rank0 med`:rank0 的调用返回时刻中位。MOSAIC 的返回 = **门开时刻**
    (recvbuf 就绪即开,与训练语义一致);NCCL 的返回含发送侧排空。
  - `makespan`:全部 16 个 rank 完成时刻的最大值(每 rank 完成分布
    由 `[prank]` 行给出)。
- **正确性门**:接收数据内建指纹校验(失败则该格记 FAIL,不出数);
  解析时核对 16/16 rank 都在场(防单 rank 静默死产生假数字)。
- **物理地板**:热节点跨机入向字节 ÷ 197.3 Gbps(离线由同一矩阵算出)
  —— α1 = 36.3ms,α2 = 53.6ms。低于地板的数字必是假的。

### 2.2 MoE 端到端(`moe-rt/train_moe_mcore.py`)

- **模型**:Megatron-Core GPT + MoE,纯 EP(TP=PP=1,EP=world=16),
  32 专家 top-2 router,4 层,hidden 512,seq 512,micro-batch 4,
  alltoall dispatcher ⇒ 每 iter 前向 12 次 + 反向 12 次 alltoallv。
- **公平性**:三样全同(模型/数据/seed/优化器),仅 `MT_BACKEND` 不同;
  mosaic 只替换 EP 的 alltoallv,DP allreduce 等其余通信保持原生 NCCL。
- **正确性门**:两后端 60 iter 的 **loss 整列 md5 逐位一致**。
- **计时**:CUDA event 包住最底层 `all_to_all_single`(前反向都必经),
  按段边界分 fwd/bwd 桶;iter/fwd/bwd/opt 墙钟分段;稳态均值(iter≥6)。

## 3. 结果

### 3.1 纯 A2A(进表 **29453543**,独占 sbatch,120 iter,全默认旋钮)

| 格 | NCCL rank0 | MOSAIC rank0 | NCCL makespan | MOSAIC makespan |
|---|---|---|---|---|
| uniform 256KB | 702.6 µs | 851.0 µs(+21%) | — | — |
| uniform 1MB | 2399.5 µs | 2793.7 µs(+16%) | — | — |
| 8MB α=0(uniform) | 17.72 ms | 18.48 ms(+4.3%) | 17.96 | 18.72 |
| 8MB α=1 | 25.74 ms | **21.09 ms(−18%)** | 40.53 | **40.53(打平)** |
| 8MB α=2 | 44.75 ms | **10.91 ms(−76%)** | 61.26 | **61.13(打平)** |

读法(如实):

- **makespan(整体完成)**:α1/α2 与 NCCL 打平,双双贴物理地板
  (61.1/53.6 ≈ 1.14×;NCCL 同为 1.14×)—— 这个维度上共享 HCA 拓扑
  没有超越空间(热节点入向是物理不变量)。
- **完成结构(MOSAIC 的差异化)**:rank0/冷 rank 的开门时刻大幅提前 ——
  α2 下 10.9 vs 44.8 ms(**4.1×**),冷 rank 7.7-34ms vs NCCL 48-57ms。
  MoE 里这意味着冷 expert 的卡可以提前开算。
- **小消息**:+16~21% 是每 collective 的固定协议成本
  (demand 交换 ~44µs + notice ~25µs + 三次 kernel 发射 + 收尾),
  消息越大占比越小;8MB 处于 ±4% 噪声带(前一次进表 29433075 为
  17.80 vs 18.08 反超,两次合并结论 = **打平**)。
- 端到端校验全过、16/16 rank 在场、零 E1025(传输重试耗尽)。
  信用反压之前,α=2 这一格是 **280.6 ms(落后 4.4×)**(29433075)。
- **2026-08-25 复核(统一搬运 API 落地后,占位 29614235 方向判断)**:
  120 iter 同尺复测 α1 24.0 / α2 11.0、uniform 8M 17.86(打平)、MoE 60it
  +11.2% loss md5 一致 —— 与本表逐字同带,本表继续有效。注意口径坑:
  **skew 格用 20 iter 短跑会把 med 落进 W-arm 探索期**(α2 会虚高到 ~79ms),
  方向判断也要 ≥120 iter 或钉 `MOSAIC_WAVES=1`(log_raven.md 08-25 结论 3)。

### 3.2 MoE 端到端(进表 **29455196** PASS,独占 sbatch,60 iter)

| 指标(稳态均值,iter≥6) | NCCL | MOSAIC | Δ |
|---|---|---|---|
| iter | 175.0 ms | 194.9 ms | **+19.9 ms(+11.4%)** |
| a2a 前向(12 coll) | 48.6 | 52.3 | +3.7 |
| a2a 反向(12 coll) | 75.4 | 85.0 | +9.6 |
| loss 整列 md5(60 iter) | b5b1d2df… | b5b1d2df… | **逐位一致** |

正确性:两后端 60 iter 的 loss 列 md5 完全相同 = 训练数值 bit-exact。
差距 +19.9ms 的逐项归因见 §3.3(无无主开销)。

### 3.3 差距归因(与 NCCL 的 +10% 花在哪,无无主开销)

nsys 双腿对照(profiling 档,rank0,12 iter;扰动 ~5%,只做归因):

| 成分 | ms/iter | 机制 |
|---|---|---|
| a2a 前向固定协议成本 | ~3.0 | 每 coll ~0.25ms(demand 广播/察觉/发射/收尾)× 12 |
| a2a 反向:方向无关结构差 | ~6.9 | 同上固定成本 + host-bounce/消费者核数据面结构 |
| a2a 反向:SM 干扰 | ~6.5 | 反向计算更密,与消费者核抢 SM 把通信自身拉长(独立旁证:计算 kernel 与通信核重叠时超出干净中位 4.2 ms/iter) |
| 段残差(fwd/bwd) | ~6.7 | 通信机器与计算的 SM/PCIe 干扰(KCONS 消融曲线的最优折中点) |

注:CUDA event 与 GPU kernel 跨度之间的"发射泡"两后端同量级
(NCCL 16.3 vs MOSAIC 13.3 ms/iter),不计入差距。
干扰类是"通信不用拷贝引擎(死锁禁用)⇒ 只能用 SM"的结构代价;
下一步优化优先级:消费者核形态 > 数据面 > 协议。

## 4. 当前方案的关键机制(数字背后是什么)

1. **接收端信用反压**(α2 悬崖的解):credit 只在接收端真正交付一片后
   铸造并回流(累计计数器,~5µs 控制面),发送端永不超过对端授权窗
   ⇒ 进水速率被排水速率闭环钳住,RNR/链路 credit 级联构造性不发生;
   demand 感知的热列初始突发钳(W0=2 片/对,含节点聚合热判据)兜住开局。
2. **W 在线自适应**:按 demand 形状签名记 completion EMA,向下探测、
   劣化档拉黑;探索下界 = 1(悬崖保护已在传输层)。uniform 自动 W=1。
3. **recv-即-开门**:门只等 recvbuf 就绪,残余发送后台排空 ⇒
   冷 rank 完成结构优势的来源。
4. **NIC 物理单元记账**:运行时探测"哪些 rank 共享同一块物理网卡",
   账记在单元上 —— 共享 HDR200 不再被当成 4 根独立管子。
5. **数据面(统一搬运 API,2026-08-25)**:搬运代码只有一份 ——
   消费者内核(SM 拷贝,4 block)是唯一搬运工,阻塞档(bench)与流序档
   (torch)同一条路;机内 CUDA IPC 直写(1-copy);跨机 verbs RC SEND,
   TX staging 池与 RX 槽池的**位置**(host pinned / 显存 dmabuf)由 init
   时 loopback QP 单向+双向带宽 profiling 每 rank 自动选定(NCCL init
   同款思路,`MOSAIC_XPORT` 可钉死)。Raven 全员自动落 pin,pin
   (GDR 读 121Gbps 根桥墙、GDR 写排水劣化都被探针识别,
   见 File/platform-limits-0821.md);本机 L40 自动落 gdr,gdr。
   门内核 + 消费者内核每 coll 起停不常驻。

## 5. 复现

```bash
# 纯 A2A 进表(Raven,独占 sbatch)
bash Impl/common/raven/rsubmit.sh a2a_16gpu.sbatch ITERS=120
# MoE 进表
bash Impl/common/raven/rsubmit.sh moe_64gpu.sbatch ITERS=60 MT_BACKENDS=nccl+mosaic
# 本机快速验证(2 主机)
cd Impl && bash regression.sh
```
