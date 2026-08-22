# TE-CCL / TACCL / SyCCL / FAST 拓扑表达能力对比(node / NIC / switch 级)

依据:三个求解器源码级调查(2026-07-08)+ dgx_rail 实战验证;FAST 部分
(2026-07-10)依据 `FAST/simulation` 源码移植(`csg-htsim/scheduler/fast_schedule.h`)
与五拓扑实测(均匀 + 非均匀,见 `2026-0710-experiment-summary-v4.md`)。
用途:拿到一个新拓扑时,判断哪些求解器能"忠实建模"它、哪些只能"降级近似"。

---

## 1. 四家的拓扑模型本质

| 方案 | 拓扑输入形式 | 一句话本质 |
|---|---|---|
| **TE-CCL** | 显式有向图 JSON(节点+边+容量+α) | "给我任意一张图,我用 LP 在上面推流" |
| **TACCL** | 主机画像 profile + 人工 sketch | "同构主机的复制品,主机内看画像,主机间听 sketch 的" |
| **SyCCL** | 参数化分层 config(hosts + layer 栈) | "标准数据中心分层结构的参数化枚举" |
| **FAST** | 只要一个分组数 G(gpus_per_server)+ demand 矩阵 | "两层 fabric 假设写死在算法里:机内快层抹平倾斜,机间 i→i 置换;不做拓扑搜索,多项式现算" |

注意 FAST 与前三家的物种差异:它不是离线求解器而是**在线调度算法**
(htsim 进程内 1–5ms 现算),不消费拓扑描述文件——拓扑假设(两层、每 GPU
独立机间通路、机内近似全交叉)是**算法的前提**而非输入。因此下面 §2 的
逐字段建模能力表对它不适用,其能力/失效面单列 §7。

---

## 2. 逐级能力表

### 2.1 Node(GPU/计算节点)级

| 能力 | TE-CCL | TACCL | SyCCL |
|---|---|---|---|
| GPU 数量任意 | ✅ 任意(图节点) | ⚠️ 必须 = 每主机 GPU 数 × 主机数(复制结构) | ⚠️ 同左(host_num × host_gpu_num) |
| 异构主机(不同主机不同 GPU 数/结构) | ✅ 图上随便画 | ❌ 所有主机是同一 profile 的副本 | ❌ 所有主机同构 |
| GPU 中继(store-forward relay) | ✅ epoch 粒度 buffer | ✅ send/recv 依赖链 | ✅ 逐跳 relay(A2A reconstruct 后 amount≡1) |
| 节点发送总量上限(node TX cap) | ✅ A2A 有 node_tx 约束(可按节点置 0 关闭) | ❌ 无显式概念(由链路 invbw 隐含) | ❌ 无(由各层链路容量隐含) |
| 非 2 幂/非整除的 GPU 组织 | ✅ | ❌ 对称约束(symmetry_offsets)实际要求规整 | ❌ 各种整除断言(见 2.2/2.3) |

### 2.2 NIC 级

| 能力 | TE-CCL | TACCL | SyCCL |
|---|---|---|---|
| NIC 是否一等公民 | ❌ 没有 NIC 概念:要么折进边容量(GPU→leaf 180G),要么建成 switch 节点 | ⚠️ 只有标量 `nics_per_node`:作用是 beta_split_factor = gpus/nics(NIC 越少跨机 β 越大),**没有布线信息** | ✅ 显式 NIC 设备:`host_nic_num` 个/主机 |
| GPU→NIC 映射 | (不适用;想表达就画边) | ❌ 不可指定(只影响带宽折算) | ⚠️ 固定连续分块:GPU i → NIC ⌊i/(gpu_per_nic)⌋,要求 gpus % nics == 0;**不能任意映射**(PerfModel.cpp:176-183) |
| 每 GPU 独立 NIC | ✅(边直接从 GPU 出) | ✅ nics_per_node = gpus_per_node(β 不缩放) | ✅ host_nic_num = host_gpu_num(gpu_per_nic=1) |
| 多 GPU 共享一块 NIC 的争用 | ⚠️ 需手工建"NIC=switch 节点"才有共享点 | ⚠️ 只体现为 β 放大,无逐 send 争用 | ✅ NIC 是共享设备,`nic_hyperedge` 约束 |
| NIC 异构(不同速率的 NIC 混布) | ✅ 边容量各写各的 | ❌ 单一 remote 画像 | ❌ 每层一个 link_spec,层内同速率 |

### 2.3 Switch 级

| 能力 | TE-CCL | TACCL | SyCCL |
|---|---|---|---|
| switch 建模方式 | ✅ 显式 switch 节点:**buffer≡0**(不存储),流量同 epoch 直通;`switch_copy=True` 时可一入多出**组播**;`switch_to_gpu_link_on=False` 时 switch→GPU 边瞬时 | ⚠️ 无 switch 节点;交换机 = "hyperedge"共享带宽约束 + 经过交换机的收发顺序约束(recv_first/send_first),主要服务**主机内** NVSwitch 语义 | ✅ 显式 switch 层设备,但内部是 "one-big-switch" 理想化(无端口级争用建模,PerfModel.cpp:373-381 注释自认) |
| 多级交换级联(leaf→spine) | ✅ 任意级联(边随便画;dgx_rail 45 节点 2 级已验证)。注意:级联图必须用 ONE_SHOT 求解 | ❌ **不存在主机间交换结构概念**:internode 是单一 remote 链路类,fabric 完全不可见 | ✅ 原生:switch 层可叠(leaf 层 + spine 层),但**spine 层只支持 pod 分组且实践上单脊**(connect_switches 非 pod 直接 assert,note:"spine 不支持 4,必须 1") |
| leaf 分组方式 | ✅ 任意(布线=画边) | ❌ | ⚠️ 仅两种模式:`multirail`(按 NIC 序号跨主机分组=轨道布线)/ `pod`(按整机连续分块);要求整除(total_nic_per_switch % machine_num == 0 等) |
| 收敛/超额认购(taper) | ✅ 上行边容量随便设,**求解器看得见 taper 并绕开它** | ❌ 看不见 spine → 只能靠 sketch(internode_conn)在结构上禁止跨轨,把"避开 spine"人工编码 | ✅ spine 层 link_spec 给低带宽即 taper(β 变大→每 epoch 容量变小),求解器可自主发现绕行(dgx_rail 实测:无提示发现 rail-relay) |
| 交换机组播(一入多出复制) | ✅ switch_copy(AllGather 有用) | ❌ | ❌ |

### 2.4 链路/异构性

| 能力 | TE-CCL | TACCL | SyCCL |
|---|---|---|---|
| 每条链路独立容量/时延 | ✅ 每条有向边独立 gbps+α,可不对称 | ⚠️ 主机内:**可以**——profile 支持任意主机内链路矩阵、每链路 invbw、并行链路重数 l(DGX-1 式 1x/2x NVLink 混布是它的看家本领);主机间:❌ 单一 remote α/β/invbw | ❌ 每层一个 link_spec,同层所有链路同速率同时延 |
| 主机内部分网(非全交叉,如环/torus/超立方) | ✅ | ✅ 链路矩阵随便写 | ❌(实测)`host_links:"nvlink"` 路径两个上游 bug:越界写(已修,PerfModel.cpp devtoGPU 缺失)+ LP 解提取 `DfsRateAlloc assert(fnd)` 失败(算法内部,未修);实现不可用,只能 nvswitch 全交叉近似 |
| 不规则/非对称全局图(如 ndv2 的 400/200/100G 混布 + 仅 2 条跨箱链路) | ✅ **只有它能** | ❌ | ❌ |
| 并行链路重数(同一对节点多条物理链路) | ⚠️ 可用多条边表达(LP 容量相加,路径不区分) | ✅ 显式 l 索引(0..link(src,dst)-1),逐链路调度 | ❌ |

### 2.5 规模与求解形态(实测)

| | TE-CCL | TACCL | SyCCL |
|---|---|---|---|
| dgx_rail(32 GPU + 13 switch 设备)A2A 2MB | 147s **OPTIMAL**(ONE_SHOT, 48×111.8us epoch) | routing 4.4s 最优 + scheduler 600s gap1.86%(**必须** relay+symmetry sketch 才可解) | **40.5s**,输出 3 个候选算法 |
| 求解形态 | 连续 LP(A2A;AllGather 是整数流) | 两阶段 MIP(routing→ordering) | MILP/LP + sketch 空间枚举,返回多算法 |
| 已知坑 | ITERATIVE(method=2)在多级交换图上系统性误报 infeasible,必须 ONE_SHOT | routing 时限原硬编码 1800s(已改环境变量);无对称 sketch 时 32 GPU 爆炸 | time_limit_h 管所有 MIP;内存峰值 ~9GB(32 GPU) |

---

## 3. 三家公共集合(都能忠实建模)

**精确刻画**:同时满足以下条件的拓扑,三家都能在各自模型里无损表达——

1. **H 台同构主机 × 每台 G 个同构 GPU**(复制结构);
2. 主机内 **NVSwitch 式全交叉**(或均匀速率的规则部分网);
3. **GPU→NIC 连续分块映射**(G % NIC 数 == 0;含每 GPU 独立 NIC 的特例);
4. 主机间是**单一速率类别的均匀连接**,且从任一主机看拓扑同构(对称);
5. 交换结构对调度的影响可以折算成"每条逻辑链路一个 (容量, α)"。

典型成员:
- `syccl_a100` / `mini_a100_16g` 那类 **主机分组 clos(pod 布线)**,只要不指望 TACCL 看见 leaf/spine 差别;
- **纯 rail 单层**(每轨一台 leaf、无 spine 或 spine 无关紧要):TE-CCL 画图、TACCL 用 direct-map sketch、SyCCL 用 multirail,全部无损;
- 单主机 NVSwitch 盒子(`syccl_a100_1host`)。

**边界情况:dgx_rail(rail leaf + 收敛 spine)属于"可执行公共集"但不是"忠实建模公共集"**——
TE-CCL 和 SyCCL 都能看见 spine taper 并自主决定避开;TACCL 的模型里 spine 根本不存在,
它能给出同样优秀的 schedule 只是因为我们用 sketch(internode_conn: direct-map)把
跨轨发送在结构上禁止了。换一个"最优解需要用一点 spine"的拓扑,TACCL 会错过那部分容量。

---

## 4. 各自独占集合(只有它能处理)

### 4.1 只有 TE-CCL 能处理

一句话:**一切不规则、非对称、异构的显式图**。

- `ndv2`(csg-htsim 版):16 GPU 无交换机异构网——400/200/100 Gbps 三种链路混布、
  两个"机箱"之间只有 2 条 100G 链路、每节点度数/邻居速率各不相同。
  TACCL 的复制主机模型和 SyCCL 的分层 config 都写不出这张图。
- `te_trap`、随机 d-regular 图、任何"研究用"人造图。
- 非对称有向容量(上行 ≠ 下行)、每条边不同 α。
- 异构主机混布(比如 3 台 8 卡 + 1 台 4 卡)。
- 需要交换机**组播**(switch_copy)语义的 AllGather 场景。
- 任意多级、任意分组、任意收敛比的交换级联(不受 multirail/pod 两模式限制,
  比如"leaf 同时按轨和按机双归属"这种混合布线)。

### 4.2 只有 TACCL 能处理

一句话:**主机内异构链路画像 + 并行链路重数 + 人工先验**。

- 主机内**每条链路速率不同**的真实画像(DGX-1:GPU 间 1×/2× NVLink 混布,
  有些 GPU 对之间 2 条并行链路、有些 1 条、有些没有)——SyCCL 的层内单速率
  写不出,TE-CCL 能画图但没有"并行链路 l 逐链路排序调度"的概念(只会容量相加)。
- **并行链路重数 l 的显式逐链路调度**(哪个 chunk 走第几条 NVLink)。
- 从实测 profiling(α-β 拟合)直接生成拓扑输入的工作流,以及
  **用 sketch 注入人类结构先验**来剪枝搜索空间——这既是限制(fabric 不可见)
  也是独有能力(可以强制 schedule 呈现指定结构,如"跨机只许同轨")。

### 4.3 只有 SyCCL 能处理

一句话:**把多层交换 fabric 当一等公民做自动结构搜索**。

- **无人工提示**下的多层 fabric 结构发现:同一 config 里给出 NVLink 层、NIC 层、
  multirail leaf 层、taper spine 层,求解器自己决定"每一步走哪一层"
  (dgx_rail 上自主发现 rail-relay、零 spine,是三家里唯一不需要任何提示做到的;
  TE-CCL 也能"发现"但要 147s LP,SyCCL 40s 且同时给 3 个候选算法)。
- `multirail` / `pod` 两种 leaf 布线的**参数化一键切换**(对比实验友好:
  同一物理清单,改一个词就从轨道布线换成整机分组)。
- **一次求解返回多个候选算法**(按预测完成时间排序,带 time_diff_threshold
  去重)——做算法多样性/鲁棒性分析只有它原生支持。
- NIC 作为共享设备的 hyperedge 争用 + 分层剪枝(ignored_layers /
  must_contain_layers / layer_comm_uplimit)这种"按层"的搜索控制。

---

## 5. 新拓扑判定 checklist(按顺序问)

1. **主机同构吗?**(每台主机 GPU 数、内部互连、NIC 数都一样?)
   否 → 只有 TE-CCL。
2. **主机间连接是不是"不规则图"?**(链路速率不一、非对称、稀疏怪异布线)
   是 → 只有 TE-CCL。
3. **有多级交换且收敛比(taper)影响最优解吗?**
   是 → TE-CCL(显式图)或 SyCCL(层栈);TACCL 只能靠 sketch 绕开、不保证最优。
4. **leaf 布线是 multirail 或 pod 之一、spine 单台、各处整除吗?**
   是 → SyCCL 可直接参数化;否 → SyCCL 出局(或降级近似),用 TE-CCL。
5. **主机内链路异构(每链路不同速率/重数)重要吗?**
   是 → TACCL(唯一逐链路建模);TE-CCL 可近似(容量相加),SyCCL 不能。
6. **GPU→NIC 映射规则吗?**(连续分块)
   否 → SyCCL 出局。
7. 都通过 → 公共集合,三家都上,才是干净的横向对比
   (注意第 3 条的 dgx_rail 式保真度脚注)。

---

## 6. csg-htsim 现有拓扑归类

(数字为均匀 A2A 2MB/pair completion_us;FAST 列附非均匀 zipf1.0 数字,
完整两组对比见 `2026-0710-experiment-summary-v4.md` §2/§3。)

| htsim 拓扑 | TE-CCL | TACCL | SyCCL | FAST |
|---|---|---|---|---|
| `dgx_rail`(主实验) | ✅ 忠实(45 节点显式图) | ⚠️ 可执行:sketch 禁跨轨(spine 不可见) | ✅ 忠实(multirail+pod taper) | ✅ **模型内主场**(G=8;均匀 2344.9 离 cut +4.8%;非均匀 3596.1 = 6.07× ECMP、比 fill 方案快 4.4–4.9×) |
| `ndv2`(16 GPU,异构混布) | ✅ **独占** | ❌(其 NDv2 profile 是另一种 2×8 relay 抽象,非这张图) | ❌ | ❌ 模型外(仅 2 条跨箱链路,"每 GPU 独立机间通路"崩塌;可执行但慢 56%) |
| `ndv2_32`(4×8,公共集合版 NDv2) | ✅(41 节点显式图,16691.9us,+3.6% vs cut) | ⚠️ 可解但 sketch 敏感(18123.7us):direct-map 下 routing MIP 600s 零可行解,精简为 2-relay/机箱后 276.6s OPTIMAL;32MB padded 版需"时间单位重标定+α 取整"编码才可解(两坑确诊见 §7 注) | ⚠️ 可执行非忠实(16110.3us,轮询选包):nvlink 矩阵路径崩溃 → 降级 nvswitch 全交叉近似,adapter 把非 cube 邻接的主机内 send 展开成 2 跳 | ⚠️ 半模型内(机内 cube-mesh 非全交叉,balance/dispatch 走多跳 NVLink;均匀 16528.4 居中,非均匀 24858.8 贴 chassis 入口界 +3.8%,1.92× ECMP) |
| `clos_32`(32 host/4 leaf/1 spine,2:1 taper) | ⚠️ 可执行(9463.7us,epoch 优先级映射造成瓶颈空转,见下注) | ✅(8076.73us,**与 ECMP 逐位相同**的直发方案) | ⚠️ 可执行(9256.3us,one-big-switch 模型的 224 个中继是真实额外负载) | ⚠️ 退化(G=1 纯 Birkhoff;均匀 9197.3 有 stage 气泡,非均匀 42439.2 与所有人同贴 hot-host 物理界) |
| `regular_32`(32 节点 d=4 直连) | ✅ 原生(4176.9us,1.156×) | ⚠️ 单节点 32-GPU links 矩阵编码(4707.8us;routing 14s OPTIMAL,证明该编码在 32 规模可解) | ❌(nvlink 矩阵路径坏、torus 未实现) | ❌ 模型外(置换配对非邻居→逐 stage 多跳中继;均匀慢 67%,非均匀慢 21%) |

**clos_32 注**:TE-CCL 的 schedule 本身是零中继纯直发、负载与 ECMP 逐字节相同;
慢 17% 是 htsim runtime"每包严格 min-epoch 优先级"在单端口混流拓扑上造成
瓶颈链路空转 ~1.2ms(LP 把同 leaf 流量排在早 epoch)。轮询选包实验证实可修
(8073.9=ECMP)但会摧毁 TACCL 的流水顺序(dgx_rail +59%),故选包语义按求解器
分治(SyCCL 轮询、TE-CCL/TACCL min-epoch),本异常作为已确诊映射代价保留。regular_32 的 basic(3638.5us)最快属预期:直连图上包级动态喷洒 +
cut-through 本就是强 baseline。

(注:syccl_a100 / mini_a100_16g / syccl_a100_1host / te_trap 四个旧拓扑已按用户要求删除。)

**ndv2_32 的设计说明**(2026-07-08,`topo/ndv2_graph.h`):为把 32-GPU NDv2 放进
三家公共集合做了两处刻意妥协——机箱内 NVLink 统一 200Gbps(保留 hybrid
cube-mesh 邻接,放弃 400/200 异构,迁就 SyCCL 层内单速率)、跨机箱走
2×100G NIC + 单台 IB 交换机(放弃稀疏点对点直连,迁就 SyCCL 的 NIC+switch
层机制)。对称 A2A 下所有方案同受机箱出口 cut(15.36ms)约束,该拓扑用于
公共集合可执行性验证,不用于展示 solver-vs-ECMP 差距(那是 dgx_rail 的职责)。

---

## 7. FAST(在线调度器)的拓扑能力与失效面

FAST(arXiv 2505.09764)不做拓扑搜索:算法结构 = sender balance(机内)→
server 级矩阵 Birkhoff 分解成置换 stage(机间,i→i 同序号配对)→ receiver
dispatch(机内),htsim 集成见 `csg-htsim/scheduler/fast_schedule.h`。
拓扑对它只是一个分组数 G(`-fast_gpus_per_server`,默认 dgx_rail/ndv2*=8、
clos/regular=1)。

**模型内(能赢)的判定条件**——三条都要成立:
1. **两层结构**:存在明显快于机间层的机内互连(NVLink vs NIC 一个数量级);
2. **每 GPU 独立机间通路**:i→i 置换要求同序号 GPU 对之间有不经共享瓶颈的
   通路(rail 布线是完美匹配;ndv2_32 的 2 NIC/箱共享是弱化版,仍可);
3. **机内近似全交叉**(NVSwitch):balance/dispatch 假设机内任意搬运便宜;
   cube-mesh 这类稀疏机内网靠 BFS 多跳仍可执行,代价可控。

**独有能力**(四家中唯一):
- **原生非均匀 alltoallv**:demand 矩阵直接进 Birkhoff,无需 padding;
  receiver-side balance 把 incast 从"热 GPU 的 NIC"摊到"热 server 的全部 NIC"
  (dgx_rail 非均匀 6.07× ECMP、贴 server 级入口下界 +8.1%,是唯一显著
  低于 hot-GPU NIC 直发界的方案)。
- **在线现算**:1.3–5.3ms(htsim 进程内实测),比求解器快 4–6 个数量级;
  倾斜每变一次求解器要按新 pad 尺寸重解(TACCL ≈14 分钟/次且可能
  Infeasible),FAST 零重解成本。

**失效面**(实测边界,均匀/非均匀结论一致):
- 非两层直连图(regular_32):置换配对多为非邻居 → 逐 stage 多跳
  store-forward,均匀慢 67%;
- 机间通路极稀疏(ndv2_16,全箱共 2 条跨箱链路):8 路 i→i 全部挤同两条
  链路,慢 56%;
- 无机内层(clos_32,G=1):退化为纯 Birkhoff 置换排序,均匀下 stage 顺序
  造成 TE-CCL 式瓶颈气泡(+14%),非均匀下与所有方案同贴物理界(无害)。
- 均匀负载本身:FAST 论文自认无优势场景;dgx_rail 实测 2344.9us,夹在
  TACCL(2432.9)与 TE-CCL(2308.2)之间,不落后即达标。

**§5 checklist 增补一问**:0. *这是不是"两层 fabric + 每 GPU 独立机间通路"
且负载倾斜/动态?* 是 → 先试 FAST(免求解);否 → 走原 1–7 判定选求解器。

**TACCL padded 版 Infeasible 确诊与修复**(2026-07-10):把 2MB 画像的
β/invbw 直接 ×16 编码 32MB padded 拓扑时,ndv2_32 在 routing 阶段、
regular_32 在 scheduler 阶段被 Gurobi **证明** Infeasible(IIS 显示卡在
TACCL 硬编码的时间哨兵上:routing `ST=5e6`、scheduler `ST=5e5`,
`time <= ST-1`;β×16 后 makespan 模型单位越界)。这不是拓扑不可解,是
**模型时间单位越界**。第一版修复(β 保持 2MB 数值、α÷16)引入小数系数
(ndv2_32 α=0.4375),配合 TACCL 的 1e-9 容差使 routing MIP 1800s 零可行解
——第二坑。最终修复(不动求解器代码):**时间单位重标定 + α 取整**
(`script/common.sh: make_padded_taccl_topo`;α 失真 <0.1us,相对毫秒级 β
可忽略),ndv2_32(routing 有解 + scheduler 600s gap 51%,htsim 35851.9us
PASSED)与 regular_32(17078.2us PASSED)均解出。dgx_rail/clos_32 的 ×16
编码恰好没越界,与重标定编码数学等价。教训:TACCL 画像标定大 chunk 时
缩 α 而非放大 β,且保持系数为整数。细节见 `2026-0710-experiment-summary-v4.md` §3/§4。

---

## 引用(源码依据)

- TE-CCL:`TE-CCL/teccl/formulations/{base_formulation,alltoall,allgather}.py`
  (switch buffer≡0:allgather.py:165-168;switch_copy:allgather.py:212-227;
  epoch/α/β 折算:base_formulation.py:45-122;node_tx:alltoall.py:275-286)
- TACCL:`taccl/taccl/{topologies/topology.py,routing.py,scheduler.py,serialization.py}`
  (链路重数 l:topology.py:161-162,264-271;relay/sketch:routing.py:241-308;
  对称压缩:routing.py:216-238;switch 顺序约束:scheduler.py:372-431)
- SyCCL:`syccl-grb/src/PerfModel/PerfModel.cpp`
  (GPU→NIC 分块:171-194;multirail:205-225;pod:226-251;spine 仅 pod:255-277;
  one-big-switch 理想化:373-381)、`src/Sketch/parse.cpp`、
  `src/Solver/ProcessTopology.cpp:62-221`(β/epoch 容量)
- FAST:`FAST/simulation/{algorithm/all2all.cpp(Birkhoff/匈牙利),
  scheduler/local.cpp(balance/dispatch), scheduler/global.cpp(pipeline)}`;
  htsim 移植:`csg-htsim/scheduler/fast_schedule.h`(FastPlanBuilder);
  论文精读:`Note/paper_fast_summary.md`
- 实战数据:`csg-htsim/topo/DGX_RAIL_STATUS.md`;
  各求解器流程 runbook:`TE-CCL/HTSIM_SCHEDULE_GUIDE.md`、
  `taccl/HTSIM_SCHEDULE_GUIDE.md`、`syccl-grb/HTSIM_SCHEDULE_GUIDE.md`
