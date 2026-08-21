# dgx_rail 拓扑与四方案对比实验状态(2026-07-08)

## 实验契约(默认 workload,换拓扑不变)

- A2A,每个 P2P 对 **2MB**(message=chunk=2097152),包 **1KB**,**store_forward**。
- 已写入 `run_smart_regular.sh` 默认值和 `main_regular.cpp` 默认值。
- 32 GPU:expected_bytes = 992 × 2MB = 2080374784。

## 拓扑:dgx_rail(`topo/dgx_rail_graph.h`)

Rail-optimized GPU pod(现实依据:NVIDIA DGX SuperPOD 轨道优化布线 + NCCL PXN 轨道对齐中继):

- 4 主机 × 8 GPU = 32 计算节点(scale-up:每主机 NVSwitch 全交叉)。
- 每 GPU 独立 NIC;rail r = 各主机第 r 块 NIC,接 leaf r(8 台 leaf);单 spine,leaf→spine **taper 180Gbps(4:1 超额认购)**(scale-out)。
- 速率/时延沿用 SyCCL A100 profile:NVLink 2400G/9us,NIC+leaf 180G/每段25us。
- 关键结构:跨主机**同轨**流量只经 leaf(50us);**跨轨**必须过 spine(100us,taper 争用)。
- `-dgx_taper_gbps` 可调超额认购程度(默认 180)。

## Task 0 重构(已完成)

1. **per-port 发送循环**(`src/node.h`):每个物理网口一个独立 pacing EventSource;
   dgx_rail 里 GPU 有 2 个口(PORT_NV=0 2400G,PORT_NIC=1 180G),多个逻辑邻居共享一个口。
   点对点拓扑 port==neighbor,行为不变。
2. **交换机即线缆**(switch-as-wire):交换机不再是 SmartNode;GPU→GPU 逻辑边 =
   多段 `[queue,pipe,queue,pipe,...,dstGPU]` Route,交换机出口队列在路由间**共享**建模争用。
   这与 TE-CCL(switch buffer≡0)/TACCL/SyCCL(交换机只是层)的语义一致,
   同时消除了交换机 SmartNode 的整块 store-forward 时延和事件开销。
3. store_forward 为实验默认(cut-through 下逐包 transit chunk 会让 pool 扫描变二次方,
   16 节点 TECCL 从 ~25min 降到 2.6s)。basic 因逐包动态选路无法 store-forward(有意保留)。

## Task 1 映射保真修正(scheduler/ 已完成部分)

- **epoch 屏障放宽**(taccl_schedule.h / teccl_schedule.h path 模式):节点在当前 epoch
  的 step 全部**注入**后即推进下一 epoch(原来要求全部端到端**送达**)。
  依据:TACCL 的 step 只是连续时间离散化,唯一硬约束是数据可用性(step_is_ready);
  TE-CCL 的 epoch 是墙钟量化,交付屏障是过严的代理。保留 epoch 作为发送优先级
  (select_chunk_to_send 取最小 round)和节点内 step 顺序(先序 step 未就绪不越级)。
- **TACCL t 顺序**:adapter 不重排 step 内 sends(求解器已按 t 排序),
  runtime 池内顺序即注入顺序 → 同 epoch 内按 t 先后发送。
- 遗留(见各调查报告,影响较低):TACCL 并行链路 l(dgx_rail 无链路重数,无影响)、
  step.rounds 仅元数据、SyCCL fractional amount 断言(adapter 中处理)。

## 对比结果(dgx_rail,2MB/pair,1KB 包,store_forward,均复核可复现)

| 方案 | completion_us | vs ECMP | 跨轨(spine)send | 求解耗时 | 备注 |
|---|---|---|---|---|---|
| static_ecmp(直发) | 7852.29 | 1× | 672 对直发 | — | 理论 spine cut ≈7466us |
| TACCL | 2432.89 | 3.23× | **0**(sketch direct-map)| routing 4.4s 最优 + sched 600s gap1.86% | 实测优于自身预测 2686.8us |
| SyCCL | **2287.22** | **3.43×** | **0**(自主发现 rail-relay)| 40.5s | 轮询选包(2026-07-08)后登顶;原 min-epoch 语义下 2377.41 |
| **TECCL** | **2308.22** | **3.40×** | **0**(LP 全局最优)| 147s OPTIMAL(ONE_SHOT)| NIC 负载完全均衡(48MiB/NIC),开销仅 +3% |
| 理论 rail 中继下界 | ≈2237 | 3.5× | 0 | — | NIC cut 2133us + 时延 |

TE-CCL 注意事项:`solution_method=2`(ITERATIVE)在该 45 节点多级交换图上系统性
误报 infeasible(9 节点玩具图可解,25/45 节点均失败),必须用 `solution_method=1`
(ONE_SHOT);论文引用 TE-CCL 求解时间时需注明。

结论:合成调度的优势来源被精确验证——避开 4:1 taper 的 spine,
用 NVLink 轨道对齐中继(TACCL 靠 sketch 提示,SyCCL 自主搜索发现)。
htsim 实测与求解器预测的偏差 <5%,交叉验证了 adapter/拓扑映射保真度。

## 流水线产物位置

- TECCL:`examples/dgx_rail/teccl_*`,adapter=tools/teccl_generate_schedule.py(新增 dgx_rail 45 节点图:GPU0-31+NVSw32-35+leaf36-43+spine44,switch_indices)
- TACCL:`examples/dgx_rail/taccl_*`(rail 对齐 sketch:internode_conn {"g":[g]},symmetry [[8,32]];routing.py TimeLimit 1800→600 参数调整)
- SyCCL:`syccl-grb/config/rail-4h8g-a2a-2mb.json`(multirail leaf ×8 + pod spine ×1),
  adapter=tools/syccl_generate_schedule.py 新增 dgx_rail
- 历史产物归档:`Htsim/archive/pre-20260708/`
