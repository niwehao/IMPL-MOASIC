# TE-CCL → htsim Schedule 合成流程指南

面向后续模型/工程师的 runbook:给定 csg-htsim 里的一种拓扑,如何用 TE-CCL
求解生成 schedule 并在 htsim 中执行对比。以 `dgx_rail`(32 GPU rail-optimized)
为已验证的工作示例。

## 0. 前提与不变约束

- conda env `myenv`(teccl CLI 可用),Gurobi 学术 license 有效。
- **默认 workload 契约**:A2A,每个 P2P 对 2MB(chunk_size_bytes=2097152),
  包 1KB,htsim 用 `-store_forward`。换拓扑不变。
- **拓扑一致性**:TE-CCL 求解用的图必须和 htsim 执行拓扑语义一致
  (链路速率、时延 alpha、交换机位置)。htsim 侧交换机是"线缆"
  (共享队列+pipe),TE-CCL 侧交换机是 buffer≡0 的 switch 节点——两者语义等价。
- 单次求解 <10 分钟(workflow 里 `time_limit_hours` ≤ 0.15)。
- 不修改 TE-CCL 求解算法本身;只允许改输入(topo/workflow json)。

## 1. 在 adapter 里定义求解图

编辑 `csg-htsim/tools/teccl_generate_schedule.py` 的 `build_graph()`,
新增一个 `name` 分支,返回带以下字段的 dict(参考现成的 `dgx_rail` 分支):

```json
{
  "format": "...", "topology_model": "nic_graph",
  "node_count": <GPU数+交换机数>,
  "endpoint_indices": [0..GPU数-1],
  "switch_indices": [GPU数..总数-1],
  "directed": true, "rate_unit": "gbps", "alpha_unit": "seconds",
  "node_tx_gbps": [...], "edges": [[src, dst, gbps, alpha_sec], ...]
}
```

要点(dgx_rail 的经验):
- GPU 编号必须与 htsim 拓扑一致(0..31);交换机放在 GPU 之后
  (NVSwitch 32-35, leaf 36-43, spine 44)。
- **alpha 与 htsim pipe 时延精确对齐**:有向边可不对称
  (如 GPU→NVSwitch 9e-6、反向 0),使"GPU 经交换机到 GPU"的总 alpha
  等于 htsim 对应逻辑边的总 pipe 时延(dgx_rail:intra 9us、同轨 50us、跨轨 100us)。
- **交换机节点 `node_tx_gbps` 置 0** 跳过 TE-CCL 的节点级 TX 上限,
  否则 leaf 的聚合转发能力会被单节点上限扼杀。
- 写出 `htsim_node_count`(= GPU 数)让 .sched 的 node_count 是 32 而不是 45。

## 2. 写 topo/workflow 配置

`csg-htsim/examples/dgx_rail/` 下的现成示例:
- `dgx_rail_topo.json`:`{"name": "dgx_rail", "node_count": 32}`
- `teccl_dgx_rail_a2a_2mb_workflow.json`:关键字段
  - `"collective": "alltoall"`, `"chunk_size_bytes": 2097152`, `"num_chunks": 1`
  - `"solution_method": 1` —— **必须 ONE_SHOT,见第 5 节陷阱**
  - `"epoch_multiplier": 16`(粗 epoch,控制 LP 规模;epoch=chunk过最快链路时间×倍数)
  - `"num_epochs"` 初始值(如 48;生成器会自动倍增重试)
  - `"time_limit_hours": 0.15`

## 3. 求解 + 生成 .sched

```bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
python3 tools/teccl_generate_schedule.py \
  --topo examples/dgx_rail/dgx_rail_topo.json \
  --workflow examples/dgx_rail/teccl_dgx_rail_a2a_2mb_workflow.json \
  --output examples/dgx_rail/teccl_dgx_rail_a2a_2mb.sched
```
结果缓存于 `cache/teccl_schedules/<hash>.*`(内容寻址,改任何输入自动换 key);
`<hash>.meta.json` 里有 gurobi 状态、epochs、耗时,先看它再下结论。

## 4. htsim 执行与验收

```bash
./sim/tests/htsim_smart_regular -nodes 32 -topology dgx_rail -packet_size 1024 \
  -scheduler teccl_schedule -schedule_file examples/dgx_rail/teccl_dgx_rail_a2a_2mb.sched \
  -workload A2A -all_to_all -store_forward -quiet
```
验收标准:
- `validation PASSED`,`expected_bytes` = 992×2097152 = 2080374784;
- `queue_drops 0`;
- 检查 .sched 里 hop 的 src/dst 全部 < 32(GPU id);
- 按需统计跨轨字节(好的 schedule 在 dgx_rail 上 spine 流量应≈0)。

## 5. 已知陷阱(实测教训)

1. **`solution_method=2`(ITERATIVE)在多级交换图上系统性误报 Infeasible**
   (9 节点玩具图可解,25/45 节点全失败,且每次失败把 chunk 分裂×32、
   建模 15-30 分钟)。**一律用 `solution_method=1`(ONE_SHOT)**:
   dgx_rail 45 节点 48 epochs 147 秒 OPTIMAL。
2. epoch 太细 → LP 变量爆炸;太粗 → 贴着理论下界的 epoch 数不可行。
   用 `epoch_multiplier` 调:先估完成时间下界,让 epochs≈下界/epoch_duration 的
   1.5-2 倍起步。
3. A2A 输出解析走 "8-Chunk paths"(hop-level);htsim runtime 只认 GPU 端点,
   "via switches" 的具体交换机路径被折叠——htsim 的逻辑边已按主机/轨关系
   唯一确定物理路径,无歧义(单 spine 时)。多等价路径的 fabric 需先扩展
   逻辑边选择逻辑再用。
4. htsim runtime 语义:epoch 是发送优先级+节点内顺序,不是墙钟;
   唯一硬约束是 chunk 可用性(收到才能转发)。这是全部三家求解器统一的执行语义。
