# FAST 运行时 stage 同步的复现与保真度审计(33 组)

2026-07-28。归档:`csg-htsim/experiments/mosaic_x1/33_fast_sync/`
(results.csv + plot.png + 三份批日志)。代码改动:
`csg-htsim/scheduler/fast_schedule.h`、`csg-htsim/scheduler/teccl_schedule.h`、
`csg-htsim/src/main_regular.cpp`。相关文档:`Note/Design/baseline_fast.md`
(旋钮速查)、`Note/SOTA.md`(2400 行更新)。

## 0. 问题与结论(一段话)

**问题**:htsim 里复现的 FAST 此前只保留了 chunk availability 这一条
硬依赖,官方实现里执行期的 stage 间同步(per-rank stage 串行、接收端
credit 反压、dispatch 等 stage 收齐)完全缺失——即 async 执行是对
FAST 有利的上界近似,且无法量化偏差。**做法**:忠实移植官方两种同步
语义(真机 kernel 的 credit 协议、simulation 分析模型的全局屏障)为
`-fast_sync {none|credit|barrier}` 旋钮,默认 none 保持历史行为逐位
可复现;同时审计整个移植,发现并修正一处影响公平的保真度问题
(stage 执行序应为 largest-freq-first)。**结论**:忠实 credit 语义下
FAST 在 zipf 主线只慢 2.3~3.6%(强 NV 2400 例外:+11.5%),uniform
主场 ≤4.3%——既有对比表引用 async 数字的方向是保守的(高估对手),
所有既有结论方向不变;但 stage 序修正让 async FAST@2400 提速 3.5%
(3541→3419),SOTA 该点 vs FAST 从 1.04× 收窄到 1.004×,已如实更新。

## 1. 官方源码里的运行时同步到底是什么

### 1.1 真机 NVSHMEM kernel(`FAST/nvidia/src/fast_alltoall/flash_alltoall_nvshmem.cu`)

官方**不是**全局 barrier(主循环里的 `nvshmem_barrier_all` 全部被注释
掉),而是四层 pairwise 门控:

| 门 | 机制 | 源码位置 |
|---|---|---|
| (a) balance 屏障(本 rank) | balance 收齐(`wait_balance_complete`)+ memcpy 进 send_buffer 后,event0 才放行 stream1 的首个 inter kernel | :2544-2557 |
| (b) per-rank stage 串行 | 各 stage 的 inter kernel 顺序排在 stream1 上,kernel 内 `nvshmem_quiet()` 保证 stage p-1 的 put 远端完成后 stage p 才开始 | :1717-1730 |
| (c) 接收端 credit(双缓冲深度 2) | sender 发 stage p 前 `wait_and_consume_credit`;接收端 `grant_credit_to_peer` 只有在 ①自己 stage p-1 的 inter 数据收齐(`sync_signal >= recv_sz`)②(p≥2)stage p-2 的 redistribute 已完成、ping-pong buffer 释放 之后才授信 | :1645-1689,初始 credit :2515 |
| (d) dispatch 门(本 rank) | stage p 的 redistribute 要等 `wait_stage_ready`(stage p 数据**全部**到达)且在 stream2 上排在 intrinsic A2A 与 stage p-1 的 redistribute 之后 | :2616,:2664-2691 |

净效果:每 rank stage 严格串行,与对端最多 1~2 个 stage 的超前窗口,
接收端有真实反压。

### 1.2 simulation 分析模型(`FAST/simulation/scheduler/global.cpp:150-212`)

更强的全局屏障:时间按段累加
`t += max(balance) → t += max(inter₀, intrinsic) → t += max(interₚ,
dispatchₚ₋₁) → t += max(dispatch_{P-1})`,即段与段之间是全局
barrier,段内 inter 与上一段的 dispatch 重叠。

### 1.3 我们此前缺什么

htsim 移植(path 模式 `SmartTecclScheduleRuntime`)只有:chunk chain
内部依赖(`step_is_ready`,`teccl_schedule.h`)+ min-epoch 发送优先级
(软)+ 节点 epoch 推进(条件是"发出"而非"送达")。四层门全部缺失,
仅剩一个副作用式约束(节点作为 landing GPU 的 dispatch 未 ready 时
挡住自己 epoch 推进,≈1 stage 松窗口)。后果:同一时刻多个 stage 混跑,
permutation 的"无 incast"不变式在执行期可以被破坏;但也免掉 straggler
等待——偏差方向未知,需要实测。

## 2. 实现(默认行为零改变)

### 2.1 结构

- **打标**:`FastPlanBuilder` 生成每个 step 时记录
  `FastStepTag{kind∈{BALANCE,INTRA,INTER,DISPATCH}, stage}`
  (`fast_schedule.h`,与 `schedule.steps` 平行)。
- **门控**:`SmartTecclScheduleRuntime::enable_fast_sync(mode, tags,
  stage_count)` + `fast_gate_open(flow)`(`teccl_schedule.h`)。注入条件
  从 `step_is_ready(chunk 可用)` 变为 `step_is_ready && fast_gate_open`;
  所有门都只依赖**交付计数器**(在 `chunk_delivered` 更新),因此每次
  交付回调重评估即可保证推进,无需定时器;按构造无死锁(门只引用
  更早 stage / 更早段的交付事件,归纳良基)。
- **credit 模式门**(逐条对映 §1.1):
  - INTRA:本 rank balance 收齐(a);
  - INTER(stage p,src g,dst r):g balance 收齐(a);p>0 时 g 的
    stage p-1 inter 全部送达(b);p>0 时 r 的 stage p-1 inter 全部
    收齐(c 前半);p>1 时 r 的 stage p-2 dispatch 全部完成(c 后半);
  - DISPATCH(stage p,at g):g 的 intrinsic 完成 + g 的 stage p
    inter 全部收齐 + g 的 stage p-1 dispatch 完成(d)。
- **barrier 模式门**:§1.2 的段序列,全局计数器版本。
- **CLI**:`-fast_sync none|credit|barrier`(默认 none),
  `fast_plan` 行打印 `sync <mode> stage_order <order>`。

### 2.2 保真度取舍(简化与理由)

- 官方 credit 以 ping-pong **buffer** 为单位;htsim 无 per-rank buffer,
  用"stage 的全部 step 交付"作为等价的完成事件(字节数相同、粒度略细)。
- (b) 的 `nvshmem_quiet` 语义(put 远端完成)映射为"该 stage 的 inter
  step 全部 delivered"——保守方向与官方一致。
- min-epoch 选包保持不变:同步模式下门控已强制 stage 序,min-epoch 只
  在门内起作用。

### 2.3 向后兼容验证

默认(none)与全部历史归档逐位一致:zipf 8 nv 点
3541.18/5067.69/5988.27/7369.14/8746.16/10126.4/20243.8/40478.6、
uniform 3 点 2344.85/2622.38/5229.2 全部复现(21 组 + baseline 速查表);
FAST 走的 `SmartTecclScheduleRuntime` path 模式与 TE-CCL 回放同路径,
none 模式逐位复现同时证明该 runtime 改动对 TE-CCL/TACCL 回放惰性
(`_fast_sync_mode == NONE` 时门恒开、计数器不更新)。

## 3. 保真度审计:哪些地方不忠于源码、是否影响公平

对照 `FAST/simulation`(算法参考)与 `FAST/nvidia`(production)逐项
核对 `fast_schedule.h`:

| # | 项 | 结论 |
|---|---|---|
| 1 | sender balance(ceil-average target、big→small 搬运) | 字节等价(`balance_server_pair`:383 vs `local.cpp` `server2server_balance`;production `alltoall_local_scheduler.cc:155` 同 ceil-avg) |
| 2 | dispatch 贪心(每行 ≤freq、列序拉取) | 字节等价(`pull_row`:464 vs `server2server_dispatch`) |
| 3 | SDSM padding(`diff = max_sum − max(row,col)`) | 等价(`decompose`:530-551 vs `all2all.cpp:36-57`) |
| 4 | Hungarian + min-freq 提取 | 等价(每轮重建匹配,取匹配最小元) |
| 5 | α-β guard 公式(对角最大元、ratio<2) | 忠实(`guard_skip_balance`:439 vs `local.cpp:187-216`) |
| 6 | **guard 的默认状态** | **澄清**:guard 只存在于 simulation(`GlobalScheduler` 构造无条件执行);**production kernel 无 guard、恒 balance**。我们默认 off=production 语义,`-fast_alpha_beta_guard`=simulation 语义;对比义务带 guard 不变(给对手更优形态) |
| 7 | **scaling 的默认状态** | **澄清**:production 直接对原始矩阵 SDSM+分解、无 scaling(`alltoall_global_scheduler.cc:246`);我们默认 unit=1 忠实于 production,`-fast_scaling` 对应 simulation 的可选路径 |
| 8 | i→i 同 index 中继(kernel"gpu i only talks to gpu i") | 忠实(结构盲 W 系列发现的前提成立) |
| 9 | **stage 执行序** | **发现且已修**:production 按 freq 降序执行("put the largest cross-node transfer at first to pipeline with intrinsic alltoall",`alltoall_global_scheduler.cc:287-292`);我们此前按 Birkhoff 提取序。已改默认 desc,`-fast_stage_order plan` 回退 |
| 10 | 运行时 stage 同步 | **发现且已补**:即本报告 §1-2 |

**#9 为什么影响公平**:同步模式下最大 stage 放最前才能与 balance/
intrinsic 充分流水;不修的话我们会系统性高估 FAST 的同步开销。实测
nv360:credit 10658→10384(−2.6%),barrier 12402→11385(−8.2%)。
对 async(none)的影响不单调:2400 desc 快 3.5%(3541.18→3418.80),
720 desc 反慢 2.6%(5067.69→5201.67),其余 6 点 + uniform 全部逐位
相同(uniform 3 个 stage 等 freq,排序无效果)。**引用原则:async
基线按点取两序较优(对手最有利)**,即 2400 用 3418.8,720 用 5067.69。

## 4. 实验:同步 vs 不同步(flex_rail n32,zipf1.0/seed42 + uniform)

全部 validation PASSED、queue_drops 0、ecn_rate_cuts 0。zipf 9 stages,
uniform 3 stages。credit/barrier 均为 desc 序。

| workload | nv | bound | none(async) | credit | barrier | credit 开销 | barrier 开销 |
|---|---|---|---|---|---|---|---|
| zipf | 2400 | 3327 | **3418.8** | 3812.3 | 3862.4 | **+11.5%** | +13.0% |
| zipf | 720 | 4532 | 5067.7¹ | 5321.1 | 5816.9 | +2.3% | +11.8% |
| zipf | 600 | 5229 | 5988.3 | 6203.9 | 6811.1 | +3.6% | +13.7% |
| zipf | 480 | 6181 | 7369.1 | 7600.3 | 8314.5 | +3.1% | +12.8% |
| zipf | 420 | 6798 | 8746.2 | 8984.3 | 9843.8 | +2.7% | +12.6% |
| zipf | 360 | 7556 | 10126.4 | 10384.1 | 11384.9 | +2.5% | +12.4% |
| zipf | 180 | 11330 | 20243.8 | 20728.9 | 22670.8 | +2.4% | +12.0% |
| zipf | 90 | 15105 | 40478.6 | 41430.7 | 45242.7 | +2.4% | +11.8% |
| uniform | 2400 | — | 2344.9 | 2445.0 | 2445.0 | +4.3% | +4.3% |
| uniform | 360 | — | 2622.4 | 2722.5 | 2722.5 | +3.8% | +3.8% |
| uniform | 90 | — | 5229.2 | 5257.4 | 5258.1 | +0.5% | +0.5% |

¹ 720 引 plan 序(该点 async 两序中较优);开销百分比相对本点 desc 序
none(5201.7)。

### 4.1 读数

1. **credit(忠实真机)开销小而稳**:zipf 90→720 落在 +2.3~3.6%,
   随 NV 变弱单调递减(NIC 越慢,stage 传输时间越长,同步气泡占比
   越小)。方向证实:**async 历史数字确实高估 FAST,但幅度不足以
   改变任何既有对比结论**(MOSAIC 侧领先幅度只会更大)。
2. **zipf@2400 是唯一大开销点(+11.5%)**:NVLink 2400 时
   balance/dispatch 近乎免费,async 下 9 个 stage 的 NIC 传输完全
   并发、瓶颈 NIC 连续满速;credit 强制 stage 串行后,每个 stage 要
   等最慢 server 的收发齐,9 个不等 freq 的 stage 串起来暴露了
   permutation 间的负载不均。这正是 FAST 论文附录最坏界
   (gap ≤ (B2/B1)(m+m/n))描述的 regime——同步开销随 scale-up:
   scale-out 带宽比增大。
3. **barrier(分析模型)≈ +12~14%(zipf)**:全局屏障把每 stage 的
   straggler 等待全部计入,是同步代价的上界;官方分析模型用它自估
   "extra overhead",与我们的包级测量方向一致。
4. **uniform(FAST 主场)同步几乎免费且 credit≈barrier**:3 个等
   freq permutation,每 stage 全网同时完成,pairwise 门与全局屏障
   等价(逐位或 ±0.7µs);nv90 时 NIC 串行时间完全掩盖同步(+0.5%)。
   FAST 论文"平衡负载下无 straggler"的自洽性在包级复现下成立。
5. **对 MOSAIC 对比的影响**:唯一实质变化是 SOTA 2400 行
   (MOSAIC C6+R 3405 vs FAST 3418.8,1.04×→**1.004×**,对我们
   不利,已如实更新);其余 7 个 rail 主线点 FAST 引用值不变或按
   较优序保持。若按忠实 credit 语义对比,MOSAIC 全线领先幅度反而
   扩大 2~12%,我们**不采用**(坚持引对手最强合法形态)。

## 5. 使用约定(今后实验)

- **对比表继续引用 `-fast_sync none`**(对手最有利;且真机的
  credit 深度依赖 buffer 尺寸等未建模工程参数,none 是干净上界);
  需要引用同步敏感性时注明 credit/barrier。
- `-fast_stage_order` 新默认 desc;**复现 ≤32 组归档的 FAST 数字用
  `-fast_stage_order plan`**(仅 2400/720 两点有差异)。
- guard 义务不变(`-fast_alpha_beta_guard`);新增旋钮全部正交,
  与 `-fast_rr_select`/`-fast_scaling` 可叠加。

## 6. 局限

- credit 的"buffer 释放"以 stage 全部 step 交付近似,比真机 per-byte
  signal 粗;真机 stage 内是单笔大 put,我们是多 chain 并发——stage
  内部完成序不同,但 stage 边界语义一致。
- 未建模调度计算本身的 μs 级耗时与 credit 信号的网络延迟(真机
  nvshmemx_signal_op 也走 NIC);两者都对 FAST 不利,忽略仍是保守
  方向。
- barrier 模式只用于上界对照,不对应任何真机执行方式。
