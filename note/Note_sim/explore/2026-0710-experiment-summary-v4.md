# htsim × 合成调度(TE-CCL / TACCL / SyCCL / FAST)实验总结

更新:2026-07-10(新增 FAST 集成 + 非均匀 A2A 全对比)。此前版本:2026-07-08。
数字均为 htsim 实测(validation PASSED、字节数核对、零丢包),可用文中命令复现。
配套文档:`2026-0708-solver-topology-capabilities.md`(能力对比,含 FAST)、
`csg-htsim/topo/DGX_RAIL_STATUS.md`(dgx_rail 详录)、`paper_fast_summary.md`
(FAST 论文精读)、三个求解器目录下的 `HTSIM_SCHEDULE_GUIDE.md`(流程 runbook)。

---

## 1. 实验配置

### 统一 workload 契约

**均匀 A2A**(换拓扑不变):每个 P2P 对 **2MB**(message=chunk=2097152 B),
992 对(32 GPU)→ expected_bytes = 2,080,374,784;包 **1KB**;**store_forward** 转发。

**非均匀 A2A**(2026-07-10 新增,MoE 风格 hot-dst Zipf):
- 生成器 `tools/gen_nonuniform_a2a.py`:种子化全局热点排名,dst 权重
  1/(rank+1)^α(默认 α=1.0, seed=42),**每个 src 的总发送量与均匀契约完全相同**
  ((N-1)×2MB),只是按权重重分给各 dst,1KB 粒度取整。
- 32 节点默认矩阵:pair 范围 0.49–18.28MB,最热 dst 收 510MB / 最冷 16MB
  (**32× incast 倾斜**),总量 2,080,374,784 B 与均匀逐字节相同 → 两组实验直接可比。
- demand 文件:`csg-htsim/cache/nonuniform/demand_n{32,16}_zipf1.0_seed42.txt`,
  htsim 经 `-demand_file` 读取。

**非均匀下均匀求解器的"填充"语义**(TE-CCL/TACCL/SyCCL 本身不支持 alltoallv):
- 计划以**最大 pair 取整到 2 的次方**为均匀尺寸求解(18.28MB → **32MB**;
  ndv2_16 10.93MB → 16MB),即 992 对每对 32MB 的均匀 A2A 计划;
- 执行时计划只当"填充容量":`tools/fill_schedule_nonuniform.py` 把每对的
  chunk/fragment 链按计划顺序截断到实际字节(src 只发实际需要的,dst 收满实际
  需求即完成,计划富余槽位从不上网)。支持全部四种落盘格式
  (teccl hop_level / teccl path / taccl_step / syccl_event)。
- FAST 与 static_ecmp/basic **原生消费非均匀矩阵**,无需填充。

### 五个方案
| 方案 | 类型 | schedule 来源 |
|---|---|---|
| static_ecmp | baseline | 固定最短路直发 |
| basic | baseline | 包级动态喷洒(直连图才有意义) |
| TE-CCL / TACCL / SyCCL | 离线求解器 | Gurobi LP/MIP,秒~十分钟级 |
| **FAST** | **在线多项式调度器** | **htsim 进程内现算(1–5ms)**,`scheduler/fast_schedule.h` |

FAST 集成方式(2026-07-10,移植自 `FAST/simulation`,arXiv 2505.09764):
sender 端 balance(机内把发往每个远端 server 的量抹平到各 NIC)→ server 级矩阵
Birkhoff/匈牙利分解成置换 stage(每 stage 每 NIC 对齐同轨 i→i 一对一发送,无
incast)→ receiver 端 dispatch(机内 NVLink 把落到中转 GPU 的数据送到最终 GPU),
三段用 epoch(balance=0,intrinsic=1,stage p 的 inter=1+p / dispatch=2+p)+
chunk 可用性依赖流水化;字节粒度,原生支持非均匀。计划落成 TE-CCL path-step
结构由现有 runtime 执行,选包用 min-epoch(保 stage 顺序)。分组:dgx_rail/
ndv2_32/ndv2 按 8 GPU/host,clos/regular 退化为 gpus_per_server=1(纯 Birkhoff)。
命令:`-scheduler fast_schedule [-demand_file F] [-fast_gpus_per_server G]`。

### 模拟器(csg-htsim)
- GPU 是唯一的 SmartNode;交换机即线缆(共享出口队列+pipe 段拼进多段 Route);
- 每物理网口独立 pacing 循环(per-port);
- 调度 runtime 语义:数据可用性是唯一硬约束(chunk 收到才能转发)。端口内
  选包语义按求解器分治:TE-CCL/TACCL/**FAST** 用 min-epoch 优先(保时序/流水/
  stage 顺序);SyCCL 用纯包级轮询。

### 求解配置(单次求解 ≤15 分钟)
| 求解器 | 关键配置 | 时限控制 |
|---|---|---|
| TE-CCL | **solution_method=1(ONE_SHOT,必须)** | workflow time_limit_hours=0.15 |
| TACCL | Custom profile + relay sketch;非均匀 padded 版把 β/invbw 按 32MB/2MB=16× 缩放 | TACCL_ROUTING/SCHEDULER_TIME_LIMIT=600 |
| SyCCL | 分层 config;非均匀 padded 版 coll.byte=32MB | time_limit_h=0.15 |
| FAST | 无求解器;htsim 启动时进程内计算 | 不适用(1.3–5.3ms) |

---

## 2. 均匀 A2A 结果(2MB/pair;FAST 为 2026-07-10 新增,其余与 2026-07-08 逐位一致)

### 2.1 dgx_rail(主对比:32 GPU rail-optimized,8 轨 leaf + 单 spine 4:1 taper)

| 方案 | completion_us | vs ECMP | 调度耗时 |
|---|---|---|---|
| static_ecmp | 7852.29 | 1× | — |
| TACCL | 2432.89 | 3.23× | routing 4.4s + sched 600s |
| **SyCCL** | **2287.22** | **3.43×** | 40.5s |
| TE-CCL | 2308.22 | 3.40× | 147s OPTIMAL |
| **FAST** | 2344.85 | 3.35× | **1.3ms**(3 stage,1664 step) |
| 理论下界(NIC cut)| ≈2237 | 3.5× | — |

FAST 与三家求解器同样收敛到"NVLink 机内换轨 + 同轨直达、零 spine"模式(它的
i→i stage 结构天然同轨),离下界 +4.8%,夹在 TACCL 与 TE-CCL 之间——但调度时间
是求解器的 **10⁴–10⁵ 分之一**。均匀负载下 FAST 论文自认无优势,此处只求不落后,
实测成立。

### 2.2 ndv2_32(4 机箱×8 GPU cube-mesh + 2×100G IB NIC/机箱)

| 方案 | completion_us | vs ECMP |
|---|---|---|
| static_ecmp | 20934.6 | 1× |
| TACCL(2-relay)| 18123.7 | 1.155× |
| **SyCCL** | **16110.3** | **1.30×** |
| TE-CCL | 16691.9 | 1.254× |
| **FAST** | 16528.4 | 1.267× |
| 理论下界(机箱 NIC cut)| ≈16100 | 1.30× | 

FAST(1.4ms)落在 TE-CCL 与 SyCCL 之间。注意 ndv2_32 的机内是 cube-mesh 而非
全交叉,FAST 的 balance/dispatch 走 BFS 多跳 NVLink 路径(模型外用法)仍成立,
因为对称 A2A 下瓶颈在机箱出口 cut。

### 2.3 clos_32(纯两层 clos:host 单上行,无机内互连;FAST 退化为纯 Birkhoff)

| 方案 | completion_us | 说明 |
|---|---|---|
| static_ecmp | 8076.73 | 贴 leaf 上行 cut(8053us)|
| **TACCL** | **8076.73** | 与 ECMP 逐位相同的纯直发 |
| TE-CCL | 9463.70 | min-epoch 映射代价(已确诊,见 4.x)|
| SyCCL | 9256.32 | one-big-switch 中继负载 |
| FAST | 9197.33 | 31 个置换 stage 的 min-epoch 排序在单端口混流下同样造成气泡 |

无 scale-up 层 → FAST 只剩 Birkhoff 置换排序,而 host 度数=1 下直发已最优,
stage 顺序约束反而引入 TE-CCL 式的瓶颈空转(9197 ≈ TE-CCL 9464 同机理)。

### 2.4 regular_32(32 节点 degree-4 直连;FAST 模型外拓扑)

| 方案 | completion_us | vs ECMP |
|---|---|---|
| static_ecmp | 4826.68 | 1× |
| **basic** | **3638.49** | **1.33×** |
| TE-CCL | 4176.90 | 1.16× |
| TACCL | 4707.76 | 1.03× |
| FAST | 8073.06 | 0.60×(**慢 67%**)|

FAST 明确的失败样本:置换 stage 的配对大多不是图邻居 → 每 stage 都是多跳
store-forward 中继 + stage 间 min-epoch 串行;"一对一无 incast"的收益在直连
稀疏图上不存在(它假设任意 pair 有独立 NIC 通路)。属 FAST 论文自述适用域
(两层 fabric)之外,保留作边界证据。

### 2.5 ndv2(16 GPU 异构混布;FAST 模型外拓扑)

| 方案 | completion_us | vs ECMP |
|---|---|---|
| static_ecmp | 6415.74 | 1× |
| basic | 6003.02 | 1.07× |
| **TE-CCL** | **5622.86** | **1.14×** |
| FAST | 10019.3 | 0.64×(慢 56%)|

2 机箱只有 2 条跨箱物理链路,FAST 的 8 路 i→i 并行发送全部挤过同两条链路的
多跳中继(每 GPU 独立 NIC 假设崩塌);单一置换 stage 也丧失流水机会。同为
模型外边界证据。

---

## 3. 非均匀 A2A 结果(hot-dst zipf α=1.0 seed=42;padded 计划 32MB,ndv2_16 16MB)

**总量与均匀逐字节相同**(每 src 62MB),只是分布倾斜(最热 dst 收 510MB,32×)。
FAST/ecmp/basic 吃 demand 矩阵;TE-CCL/TACCL/SyCCL 吃 padded 均匀计划 + fill 截断。

### 3.1 dgx_rail —— FAST 的主场(论文场景复现)

| 方案 | completion_us | vs ECMP | 对照界 |
|---|---|---|---|
| static_ecmp | 21829.2 | 1× | > hot-GPU NIC 直发界 17575us(spine 争用)|
| TE-CCL(fill)| 17450.4 | 1.25× | **≈ hot-GPU NIC 直发界**(17575us)|
| TACCL(fill)| 17451.1 | 1.25× | 同上(两家 fill 后都卡死在热 NIC 入口)|
| SyCCL(fill)| 15927.0 | 1.37× | 略破直发界(计划自带 928 个 NVLink 中继)|
| **FAST** | **3596.05** | **6.07×** | **hot-server 8-NIC 合计入口界 3327us,+8.1%** |

机理:热 dst 的 394MB 跨机流量,直发类方案全部挤进它那**一块** 180G NIC
(≈17.5ms);FAST 的 i→i + receiver dispatch 让整台 server 的 **8 块 NIC 分摊**
入口,再用 2400G NVLink 送到热 GPU——瓶颈从"热 GPU 的 NIC"搬到"热 server 的
NIC 总和"。这正是 FAST 论文的核心主张,包级模拟证实(9 stage,计划 1.8ms)。
对比:FAST 比三家求解器的 fill 方案快 **4.4–4.9×**,比 ECMP 快 6.1×。

### 3.2 ndv2_32

| 方案 | completion_us | vs ECMP |
|---|---|---|
| static_ecmp | 47768.7 | 1× |
| TE-CCL(fill)| 41234.5 | 1.16× |
| TACCL(fill)| 35851.9 | 1.33× |
| SyCCL(fill)| 41234.6 | 1.16× |
| **FAST** | **24858.8** | **1.92×**(hot-chassis 2-NIC 入口界 23956us,+3.8%)|

机箱只有 2 块共享 NIC,入口界比 dgx_rail 紧得多,但 FAST 仍把它贴住;fill 方案
慢 44–66%:padded 计划的时序/配对是按均匀负载排的,热流量在错误的 epoch 槽里
排队(TACCL 的 42 步重中继计划反而在倾斜下最耐打,但仍比 FAST 慢 44%)。
注:TACCL 的 32MB padded 画像两踩坑——β×16 编码撞时间哨兵报 Infeasible、
α 重标定成小数后 routing 1800s 零可行解——最终 **α 取整**版可解(routing 有解,
scheduler 600s gap 51%),见 §4.2。

### 3.3 clos_32 —— 无人能赢的对照组

| 方案 | completion_us |
|---|---|
| static_ecmp | 41289.6 |
| TE-CCL(fill)| 40817.9 |
| TACCL(fill)| 41289.6 |
| SyCCL(fill)| 41288.2 |
| FAST | 42439.2 |

热 host 的**单条 100G 下行**是不可绕过的物理瓶颈(界 40781us),所有方案 ±2%
贴界。证明非均匀下的调度收益同样"完全来自拓扑给中继留下的旁路空间"——clos_32
没有 scale-up 层,谁也变不出第二条进热 host 的路。

### 3.4 regular_32 / ndv2_16(FAST 模型外)

| 拓扑 | ecmp | basic | teccl(fill) | taccl(fill) | fast |
|---|---|---|---|---|---|
| regular_32 | 14748.6 | **12505.3** | 12978.0 | 17078.2 | 17911.3 |
| ndv2_16 | 8197.53 | **6484.03** | 7725.04 | — | 11264.9 |

直连/异构图上非均匀的赢家仍是包级动态喷洒(basic);TE-CCL fill 小幅优于
ECMP;TACCL 的 padded 计划(13 步单节点编码,scheduler MIP 次优)比 2MB 版
相对退步(慢 ECMP 16%);FAST 继续因多跳置换中继垫底——与均匀结论一致,
倾斜不改变其模型外劣势。

### 3.5 非均匀小结

- **两层 fabric(dgx_rail/ndv2_32)**:FAST 大胜(1.9–6.1× vs ECMP、比最好的
  fill 方案快 1.44–4.43×),且是唯一贴住 server 级入口下界的方案——
  receiver-side balance 是治 incast 的结构性手段,fill/padding 治不了。
- **fill 语义本身是干净的**:三家 padded 计划全部 validation PASSED,交付字节
  与 demand 矩阵逐字节相同;它们输在计划结构(均匀假设),不是填充实现。
- **无旁路空间的拓扑(clos)**:大家一起贴物理界,FAST 不加害(-2.8%)。
- **模型外拓扑(regular/ndv2_16)**:FAST 惩罚明显(-22%~-37% vs ECMP),
  用前必须判断"两层 + 每 GPU 独立 NIC"假设是否成立(见能力文档 §7)。
- padding 浪费的是**求解时间而非带宽**:fill 后只传输实际字节;但 TACCL 在
  32MB padded 拓扑上的 MIP 与 2MB 同规模(≈14 分钟),每换一次倾斜幅度
  (pad 尺寸变化)都要重解——FAST 现算 2ms,这是"预计算 vs 在线"的根本差距。

---

## 4. 为什么某些 Schedule 不能在某些 topo 上运行

详细逐字段分析见 `2026-0708-solver-topology-capabilities.md`;下面是实验实证到的机制。

### 4.1 模型表达力不足(根本不可行,换参数也没用)
- **TACCL / SyCCL 无法运行 16-GPU 异构 ndv2**:400/200/100G 混布 + 仅 2 条跨箱
  点对点链路。TACCL 是"同构主机复制品"模型;SyCCL 分层 config 每层单速率。
  → 这类不规则/异构图是 TE-CCL 独占。
- **SyCCL 无法忠实运行 dgx_rail 之外的任意 leaf 布线**(multirail/pod 两模式)。
- **TACCL 看不见多级 fabric**(dgx_rail 的 spine 靠 sketch 禁止跨轨绕开)。
- **FAST 不适用于非两层拓扑**(新增):它不是"不能执行"——任何连通图上都能
  生成合法计划(BFS 多跳路径)——而是**结构假设失效后性能反噬**:
  regular_32 慢 67%、ndv2_16 慢 56%(见 2.4/2.5)。判定标准:是否存在
  "快机内层 + 每 GPU 独立机间通路"。clos 类(无机内层)退化为纯 Birkhoff,
  均匀下小亏(min-epoch stage 气泡),非均匀下无害也无益。

### 4.2 求解方法在特定图上失效
- **TE-CCL ITERATIVE(method=2)在多级交换级联图上系统性误报 infeasible**,
  必须 ONE_SHOT。
- **TACCL routing MIP 对 sketch 参数极敏感**(ndv2_32 direct-map 600s 零可行解;
  2-relay 276.6s OPTIMAL)。
- **TE-CCL epoch 数下限效应**:epoch 总时长贴理论下界时必然 infeasible,
  留 1.5–2× 裕量起步。
- （非均匀新增)padded 求解是**重复成本**:pad 尺寸随倾斜幅度变化,倾斜一变
  三家全部重解;FAST 无此问题(在线现算)。
- （非均匀新增)**TACCL 大 chunk 画像的两个坑**(2026-07-10 确诊,均非拓扑不可解):
  1. **时间哨兵越界**:32MB padded 画像按 β/invbw ×16 编码时,ndv2_32(routing)
     与 regular_32(scheduler)被 Gurobi 证明 Infeasible——IIS 落在 TACCL 硬编码
     的 `time <= ST-1` 哨兵(routing ST=5e6 / scheduler ST=5e5)上。
  2. **小数系数拖死 MIP**:改用时间单位重标定(β 保持 2MB 数值、α ÷16)后
     α 变小数(0.4375),配合 TACCL 的 1e-9 整数/可行容差,ndv2_32 routing
     1800s 内零可行解(整数系数 2MB 版 276s OPTIMAL);regular_32(α=3.125)
     侥幸可解。
  最终方案:**时间单位重标定 + α 取整**(失真 <0.1us,相对毫秒级 β 可忽略),
  ndv2_32/regular_32 均解出。实现:`script/common.sh: make_padded_taccl_topo`;
  教训:TACCL 画像标定大 chunk 时缩 α 而非放大 β,且保持系数为整数。

### 4.3 runtime/adapter 层的历史偏差(已修复)
- epoch 交付屏障过严 → 已放宽为"注入即推进";
- TACCL `step.rounds` 丢弃 → epoch=rounds 累积和;
- 交换机曾是存储转发节点 → switch-as-wire;
- cut-through 逐包 transit 使模拟慢数百倍 → 契约用 store_forward;
- (新增)fill 工具首版丢零字节 fragment 后 id 有洞 → 已按出现序重编号
  demand/fragment id(TE-CCL hop 模式 runtime 要求连续)。

---

## 5. 产物索引

- **实验脚本(一键复现)**:`csg-htsim/script/run_{dgx_rail,ndv2_32,ndv2_16,clos_32,regular_32}.sh`
  + `common.sh`。均匀:直接跑;**非均匀:前缀 `NONUNIFORM=1`**(自动生成/缓存
  demand → padded 求解(缓存)→ fill → 模拟);`SCHEDULERS="..."` 挑方案,
  `FORCE_RESOLVE=1` 强制重解,`NONUNI_ALPHA/NONUNI_SEED` 换倾斜/种子。
- **FAST 集成**:`csg-htsim/scheduler/fast_schedule.h`(调度器类 + FastPlanBuilder,
  含 Birkhoff/匈牙利、balance/dispatch 的字节级移植);htsim 参数
  `-scheduler fast_schedule -demand_file F -fast_gpus_per_server G`;
  上游参考代码 `FAST/simulation/`(学术许可)。
- **非均匀工具链**:`csg-htsim/tools/gen_nonuniform_a2a.py`(demand 生成)、
  `csg-htsim/tools/fill_schedule_nonuniform.py`(四格式 fill);
  产物集中在 `csg-htsim/cache/nonuniform/`(demand、padded workflow/topo/config、
  padded .sched、`*_filled.sched`、taccl_*_32mb/ 求解目录);
  SyCCL padded config 在 `syccl-grb/config/*-32mb.json`。
- dgx_rail:`csg-htsim/examples/dgx_rail/`、`csg-htsim/cache/syccl_dgx_rail_a2a_2mb.sched`
- ndv2_32:`csg-htsim/examples/ndv2_32/`、`csg-htsim/cache/taccl_ndv2_32/`
- clos_32 / regular_32 / ndv2_16:`csg-htsim/examples/{clos_32,regular_32,ndv2_16}/`
- 拓扑源码:`csg-htsim/topo/{dgx_rail_graph.h, ndv2_graph.h(含 ndv2_32), clos_graph.h, regular_graph.h}`
- 历史产物归档 `archive/pre-20260708/` 已于 2026-07-10 删除(内容为已淘汰
  旧拓扑的求解器中间文件,无任何在用数据依赖)
