# Baseline:TE-CCL(离线 LP/MILP 求解器)

> 代码结合版总结(2026-07-12)。求解器源码在仓库外部依赖(见
> `THIRD_PARTY_REPOS.md`);本仓库侧 adapter/runtime 全部行号如下。

## 一句话

任意有向图上的 LP/MILP collective 求解器——**表达力三家最强**(唯一能画
异构不规则图),但求解最慢、pad-to-max 税最重、执行是纯回放。

## 代码位置(本仓库侧)

| 组件 | 文件 | 说明 |
|---|---|---|
| topo 描述生成 + 求解调用 | `csg-htsim/tools/teccl_generate_schedule.py` | 内容寻址缓存(输入不变即命中);dgx_rail 视图支持 `nv_gbps/nic_gbps/hosts/gpus_per_host/taper_gbps` 字段(flex 变体用) |
| `.sched` 解析 | `scheduler/teccl_schedule.h:58`(`load_teccl_schedule`) | path 模式(`step` 行,显式逐跳路径)与 hop 模式(`hop` 行,fragment 逐跳)|
| 执行 runtime | `scheduler/teccl_schedule.h:328`(`SmartTecclScheduleRuntime`) | chunk 可用性 `_available_chunks`/`step_is_ready` 是唯一硬依赖;epoch 推进 = 本 epoch 全部发出(issued)即可,不等交付 |
| 选包语义 | `teccl_schedule.h:192` min-epoch | 换纯轮转会救 clos_32 但伤 TACCL,按 solver 分治(见其 185-191 行注释) |
| 非均匀填充 | `tools/fill_schedule_nonuniform.py` | hop 模式填充后必须重编号 demand/fragment id(runtime 要求连续,工具已做) |

## 优点

- 表达力:逐边独立容量+时延、可不对称、switch 显式节点(buffer≡0)、
  组播(switch_copy)——**ndv2 16-GPU 异构混布只有它能解**;
- LP 最优性保证(均匀 collective 内);
- 弱 scale-up 时能自己发现 RX 型中继(13 组:nv=90 下 22640,好于 FAST
  的 40479)。

## 限制(报告 §1.4 的 S 项 + TE-CCL 行)

- **S1 pad-to-max**:LP 原则上可表达非均匀,但 chunk/workflow 工具链按
  均匀 chunk 构造 → 实际路径是 pad-to-max;α=2.0 时退化到 ≈ECMP;
- **S2/S4 求解规模**:147s@32 GPU,**64 GPU 30min 超时**(18 组实测);
  每换 pad 尺寸(demand 矩阵形状)要重解;
- **执行映射代价**:min-epoch 在单端口混流拓扑上可能空转(clos_32 慢
  17%,`../explore/2026-0710-experiment-summary-v4.md` §2.4,已确诊为映射代价、如实保留)。

## 坑(必读)

1. **多级交换图必须 `solution_method=1`(ONE_SHOT)**;ITERATIVE(=2)会
   系统性误报 infeasible——不是拓扑不可解。
2. **上游内置拓扑代码有 9 个已确认 bug(全部未修,按铁律不动 solver)**,
   原记录在已删除的 teccl_builtin_topologies.md,此处保存:
   - `mesh.py:19` 缺行边界检查 → "Mesh"实为柱面环绕链;
   - `amd.py:60` off-by-one(`i>16` 应为 `>=16`)→ 每箱第 0 个小交换机
     容量 50 而非 25;`amd.py:95` 连锁 α 也取错;
   - `dgx2.py:15` chassis=2 写死;`dgx2.py:24-26` α 硬编码 →
     `get_min_alpha()` 恒 0;
   - `ndv2.py:14-19` conversion_map 重复赋值(死代码);
   - `topology.py:17` equivalent_node_indices 无人赋值 → AllGather 对称
     剪枝恒不生效;
   - `regular_graph.py:24` **不写 `rate_unit:"gbps"` 速率放大 8×**
     (本仓库生成器已总是写);`regular_graph.py:34` node_tx_capacity
     仅 AllToAll 读取。
   ⇒ **只用 RegularGraph 入口 + 本仓库生成器**,别碰内置拓扑。
3. htsim 执行侧 topology 白名单在 `main_regular.cpp`(teccl_schedule 分支)
   ——新拓扑要加进去(flex_rail 是 v5 加的)。

## 怎么跑(复用优先)

```bash
# 标准流程(dgx_rail,自动缓存):
NONUNIFORM=1 NONUNI_ALPHA=1.0 SCHEDULERS=teccl ./script/run_dgx_rail.sh
# flex 变体(nv≠2400):topo json 写 {"name":"dgx_rail","node_count":32,
#   "taper_gbps":180.0,"nv_gbps":360.0} → teccl_generate_schedule.py
#   → fill → htsim -topology flex_rail -scheduler teccl_schedule
# 已有解(勿重跑):cache/nonuniform/teccl_{dgx_rail,flex360,flex90}_a2a_*.sched
```

历史成绩(zipf1.0/seed42,PASSED):nv2400=17450 / 360=11762.4 / 90=22640.3;
α 扫描与 n=64 超时见 `experiments/mosaic_x1/{13,15,18}_*/`。
