# MOSAIC(纯电 EPS 主报告):面向 MoE 非均匀 All-to-All 的 multi-corridor sub-schedule 装配与在线比例自适应

**Pre-paper 文档(v7,2026-08-05)。**

## 这份文档讲什么(先说人话)

**问题**:MoE 大模型训练/推理时,几十块 GPU 之间要互相交换数据
(alltoallv),而且流量**很偏**——个别 GPU 会收到别人十几倍的数据,
成为拖慢全局的短板;流量矩阵还每几百毫秒变一次。传统做法两头堵:
离线求解器算得好但要几分钟(等算完流量早变了),在线方法快但把
硬件假设写死(机内高速通道一弱就崩)。

**我们的方案(MOSAIC)一句话**:把每对 GPU 的数据拆到几条**走廊**
(不同的物理路径类:直发、借邻居中转、换轨道等)上,拆分比例先用
一个闭式公式按流量矩阵算个好初值,再在执行中分**波次**发送、随时
调整。生成一个调度只要几十微秒,所以每个新流量矩阵都现算现用。

**现行版本(§11 的 MOSAIC-U2)比老版本改了什么**:老版本靠"实测
各走廊吞吐、按比例学习"来调整(λ 反馈);后来审计发现——**拥堵
里九成是我们自己发的数据造成的,这部分根本不用学,记账就行**。
新核给每个物理口(网卡上下行、机内通道、交换机上行)记一本"已
分配未送达"的账,每个波次把配额填到"账面最闲"的走廊组合里;
测量只保留一件事:发现某块网卡被外界拖慢了(降速/干扰)就绕开它。
**账怎么传是老实算过的,而且改过一次口供**:不假设"所有机器瞬间
知道一切"——实测计数每 100µs 广播一次(32-64 字节,通道 A);
最初我们以为"分配决策各机自己重算"就能免传,后来自己的探针把这
条捷径测死了(视图差一拍的机器每波会把 11-39% 的字节算到别的走廊,
账越记越歪),于是分配决策也改成广播:每个目的地发波次时把"这一
波怎么分"广播出去(每波 0.3-2.6KB,通道 B),外加几微秒的决策
错峰(同时触发的机器排队再决策)。这套全可实现的协议比假设成立的
旧版只慢 **0.1%(聚合)**,而强行逐波等全局同步或完全不交换信息都
明显更差(§11.2/§11.6 有全套数字与可部署路线)。效果:对旧版 λ
方案 67 个标准测点 **39 胜 18 负**(聚合快 5%),**没人通知的网卡
降速场景快 2.15 倍**,对过期流量先验逐位免疫;负点集中在同一个
"机内带宽极弱且高度饱和"的 regime(§11.5 一张表全列并给了归因
与解法方向)。

**怎么读**:只想要当前方案 → 直接读 **§11**(设计+数字都在);想看
它是怎么被逼出来的 → §3-§4(老设计)、§10(38 组统一方案史)、
`explore/2026-0805-x41a-eps-foundation-audit.md`(为什么老根基有问题);
每个对比实验的原始数据都在 `csg-htsim/experiments/mosaic_x1/`。
**§0-§10 = 历史与依据,§11 = 现行 SOTA**;失败与负结果记录一律在
`Note/explore/` 各篇。
本文档取代原研究计划
`idea_moe_subschedule_assembly.md`(其内容已并入 §1/§2/§8/附录)。
v5 新增:全 baseline 对比(13)、均匀负载拥塞(14)、skew 扫描(15)、
corridor 数/threshold 消融(16/17)、schedule 合成速度(18)、NIC 降速
干扰(19)、host 形态(20)、设计迭代 **C=6 新默认**(21)、X2 pod/X3
双平面拓扑探索(22/23,原 §8.2 计划提前完成)、数学分析
(§4.9)与真实硬件映射(§5)。范围约定:只做 **scheduling 层面**的分析,
硬件执行细节(thread block / proxy / QP / 真实机型标定)为 out of scope。
所有进表数字 validation PASSED(交付字节与 demand matrix 逐位核对);
实验总集(25 组,每组 CSV + 图 + README):`csg-htsim/experiments/mosaic_x1/`。

---

## 0. TL;DR:faster / robust / general

**做了什么**:MOSAIC,一个在线 alltoallv scheduler。把每个 GPU pair 的流量
拆到多条 **corridor(可独立执行的 sub-schedule)** 上,每条占用不同的
物理资源;拆分比例 λ 先由**闭式 waterfilling prior** 给初值,再在执行中按
**每波(wave)实测 throughput 的反馈**动态调整。构造全程只用
pruning / greedy / closed-form,零 solver、零 LP。
实现:`csg-htsim/scheduler/mosaic_schedule.h`(单 header)。
对手:FAST(arXiv 2505.09764,最强在线 baseline,逐行移植官方库、带官方
guard;`csg-htsim/scheduler/fast_schedule.h`)+ static ECMP + 三家离线
solver(TE-CCL/TACCL/SyCCL,pad-to-max,弱侧全对比见 §6.8)。

**三支柱**:

- **Faster(合成快)**:prior 构造 20µs@32 GPU → 0.8ms@256 GPU(实测
  `mosaic_plan_us` 中位数,O(N²) 闭式);FAST 完整 plan 0.6ms→109ms
  同区间;离线 solver 同一 collective 40s–14min@32 GPU、TE-CCL 64 GPU
  即 30min 超时(§6.14)。"快速及时装配大 schedule"
  = 每个 collective 到达时按当下矩阵现算。
- **Robust(不确定性下稳)**:3 seed × 4 skew × 2 family(§6.4/6.10)、
  陈旧 prior(§6.5)、spine taper(§6.7)、**未通告的 NIC 降速**(§6.12:
  热点降速把 13:1 强侧从平手变成赢 FAST 1.44–1.50×)、buffer 压力
  (§6.6:wave = implicit flow control,ECMP/FAST FAILED 时全 PASSED)。
  反馈环是统一机制:纠 prior 误差、纠负载漂移、绕运行时干扰。
- **General(regime 宽)**:8 点 NV:NIC 扫描(§6.1/6.9)+ 弱侧全 baseline
  (§6.8)+ 均匀负载逼平(§6.6/6.11)+ host 形态 8×4/4×8/2×16(§6.13)+
  多热点/尺寸(§6.6)+ **三种布线家族**(rail / pod / 双平面,§6.15/6.16:
  corridor 族按同一剪枝流程再生成、两管 prior 公式零改动,pod 上全 9 点
  赢 FAST 1.19–2.59×)。诚实边界:强 scale-up + 温和 skew 的非主线负载下
  FAST 的全局 permutation 协调仍有 2–8% 优势(§6.9/§7)。

**设计迭代主结果(§6.9)**:corridor 数消融(§6.10)发现 SPREAD 多 lane
一致有益,**v5 默认从 C=4 改为 C=6**:主线(hot-dst zipf1.0/seed42)
**全部 8 个 NV:NIC 点同时优于旧默认,且对 FAST 全点 ≥1.00×**
(1.00×@13.3:1 → 2.49×@0.5:1)。v4 的 **crossover ≈ 3.5:1** 由此升级:
crossover 是 C=4 版本的性质;lane 加密后主线不再有输的区间。MOSAIC 全程
距 cut lower bound +7%~15%,极端 skew 处 +0.4%(§6.10)。

**三个意外发现**(v4 结论,v5 全部复验):
1. scale-up 很弱时(0.5:1),连 **uniform A2A**(FAST 的主场负载)它都输
   MOSAIC 1.17×;
2. 64 GPU 规模上 1GB fabric buffer 被热点流量冲爆——ECMP 和 FAST 丢包
   FAILED,**MOSAIC 全部通过:按 wave 发送天然限制 in-flight 字节数,
   等于免费的 flow control**;
3. FAST 对 spine taper 与热点 NIC 降速完全无感(静态 plan 无人回收错误
   估计);MOSAIC 零配置自适应。

---

## 1. Motivation

### 1.1 问题:MoE 的 alltoallv 是 skew + 动态的

MoE 训练/推理里的 `alltoallv` 占训练时间 30–56%(FAST 论文实测)。两个
特征使现有方案失效:**skew**(gating 让部分 expert 被频繁选中,热点 pair
流量可达中位数 12×,straggler NIC 拖慢整个 collective)和**动态**(流量
矩阵每几百毫秒变一次,不可预测)。

现有两类方案各有结构性缺陷:
- **离线 solver**(TE-CCL / TACCL / SyCCL):有 topology 级质量,但求解要
  秒到几十分钟(本仓库实测:TE-CCL 147s、SyCCL 40s、TACCL ~14min),对
  几百毫秒一变的负载完全来不及;且它们只解 uniform collective,非均匀
  只能 **pad-to-max**(按最大 pair 补齐成均匀再解)。
- **在线方法**:FAST 是最近也是最强的竞品(Birkhoff 分解,µs–ms 级现算,
  原生非均匀),但它把两层 fabric 的假设写死在算法里(§1.3)。

### 1.2 pad-to-max 的代价:实测,不是推演

我们先把 pad-to-max 的代价量化(dgx_rail,hot-dst Zipf,最热 GPU 收
510MB = 32× incast;完整数据 `Note/explore/2026-0710-experiment-summary-v4.md` §3):

| 方案 | completion (µs) | 对照 |
|---|---|---|
| static ECMP | 21829 | — |
| TE-CCL pad-to-max + fill | 17450 | ≈ 热 GPU 单 NIC 直发 cut(17575)|
| TACCL pad-to-max + fill | 17451 | 同上,两家逐 µs 卡在同一堵墙 |
| SyCCL pad-to-max + fill | 15927 | 计划自带中继,略破直发 cut |
| **FAST(在线,原生非均匀)** | **3596** | 热 server 8-NIC 入口 cut 3327,+8.1% |

fill 语义已把"src 只发实际字节"做到极致(padding 槽位不上网),三家
solver 仍慢 FAST 4.4–4.9×——**输的是计划结构,不是字节**:uniform 假设下
的路由/配对没有把热点入口摊开的机制。padding 还会弄坏 solver 本身
(TACCL 的时间哨兵与小数系数两坑,`explore/2026-0710-experiment-summary-v4.md` §4.2)。
v5 的关键延伸(§6.8):把 scale-up 调弱后给 solver 重解,**拓扑感知也
救不了 pad-to-max**——TE-CCL 明知 nv=90(它把 RX 型中继解了出来)仍输
MOSAIC 1.38×。结论:非均匀 A2A 的战场在**在线方法内部**,对手就是 FAST
(三家 solver 的系统性弱点清单另见 §1.4;13 组补证:即使按每个 nv 点
重解、拓扑感知到位,pad-to-max 仍让它们全点落后 1.37–4.9×)。

### 1.3 FAST 的真实结构与弱点(基于逐行移植 + 实测,非论文转述)

FAST 每个字节的链路:`origin GPU →(sender balance,NV)→ 同机 relay →
(i→i 同 rail NIC)→ 目的机同号 GPU →(receiver dispatch,NV)→ 最终 GPU`。
移植后实测纠正了我们最初的三个误判:它**能**免费复现 solver 们的
"换 rail 中继"结构(rail 布线下 i→i 天然同轨)、它**能**治接收端 incast
(实测贴 server 级入口 cut +3.8~8.1%)、它的工程极简(1.6–1.9ms 现算)。

但它有一组结构性弱点(全部实测):
- **W1 fabric 盲**:server 粒度以下拓扑不可见,从不用 spine(§6.7 taper
  实验:三种 taper 数字逐位相同);
- **W2 α 盲**:每字节强制 3 跳,小消息时绕行是纯亏;
- **W3 stage 相位屏障**:min-epoch 下 stage 间刚性排序(clos 均匀实测
  +14% 气泡);
- **W4 模型外拓扑反噬**:regular_32 慢 67%、ndv2_16 慢 56%;
- **W5 无执行期调整**:每 stage 是静态置换,估错无人回收(§6.12 的
  未通告降速实验是其代价的直接实测);
- **W6 碎 skew 的相位量化**:padding 槽占相位预算;
- **W7 scale-up 充裕性假设**(本文主攻):balance+dispatch 每字节两过
  机内层,NV:NIC 比一低就是自我加压;其源码的 α-β guard
  (`FAST/simulation/scheduler/local.cpp:187-216`)只比较 α:β、不看
  nv:nic,弱 scale-up 下 0 触发(§6.2)。

### 1.4 离线 solver 的结构性弱点(与 §1.3 的 W 项对应的 S/T 项)

FAST 不是唯一要解剖的对手。三家离线 solver 的失效方式**不同于 FAST 且
彼此不同**——W 项是"结构假设写死",S/T 项是"离线 + 均匀 + 模型边界"。
全部结论有实测或源码依据(出处逐条标注;能力边界全景见
`Note/explore/2026-0708-solver-topology-capabilities.md`,下称 CAP)。

**共性弱点(三家同有,S 项)**:

- **S1 均匀假设锁死——不适合一切非均匀 workload**:TACCL/SyCCL 的
  DSL 只解均匀 collective;TE-CCL 的 LP 原则上可表达非均匀 demand,但其
  chunk/workflow 工具链按均匀 chunk 构造(直接解非均匀 LP 留作 §8.1 的
  oracle 实验)——三家实际可用的路径都是 pad-to-max;**pad 税随 skew 超线性**
  (§6.10:α=0.5 时最好的 solver 只输 MOSAIC 1.8×,α=2.0 时三家 5.8–6.6×,
  TE-CCL 43271 ≈ ECMP 43421——"有计划"的价值被 padding 全部吃掉)。
  不适合:MoE alltoallv、专家并行、任何 per-pair 字节不等或随迭代漂移的
  通信。
- **S2 求解时延与负载漂移的量级矛盾——不适合动态 workload**:同一
  collective 40s–14min@32 GPU,TE-CCL 64 GPU 即 30min 超时(§6.14);
  且 **pad 尺寸随矩阵变 ⇒ 每个新矩阵形状都要重解**(§6.10 RUN15 实测:
  α 每变一档,TE-CCL 重解 130s、TACCL 686s)。MoE 矩阵几百 ms 一变 ⇒
  解出来时输入已过期。
- **S3 执行 = 回放,无执行期调整**:与 FAST 的 W5 同类但更僵硬——连
  stage 内的字节数都定死;taper 漂移、NIC 干扰(§6.12 类不确定性)无人
  回收。epoch 离散化还引入映射代价(clos_32:TE-CCL 计划与 ECMP 逐字节
  相同却慢 17%,min-epoch 空转 ~1.2ms,`explore/2026-0710-experiment-summary-v4.md` §2.4)。
- **S4 规模墙**:LP/MIP 变量数随节点数爆炸(TE-CCL 32→64 =
  147s→>1800s 超时;TACCL 的 sketch 在新规模要人工重设计;SyCCL 64 可解
  但 32 GPU 已 ~9GB 内存峰值,CAP §2.5)。

**各家专属弱点(T 项;"不适合什么拓扑"的正面回答)**:

| | 模型/机制缺陷 | 由此不适合的拓扑/场景 | 证据 |
|---|---|---|---|
| **TE-CCL** | T1 switch buffer≡0 + epoch 量化:表达力最强(唯一能画 ndv2 异构图)但执行映射有代价;T2 ITERATIVE 求解法在多级交换图**系统性误报 infeasible**,必须 ONE_SHOT;T3 三家中求解最慢(147s@32,64 超时) | 多级 fabric 要小心求解模式;64 GPU 以上、动态负载全不适 | CAP §2.3/§2.5;`explore/2026-0710-experiment-summary-v4.md` §4.2;§6.14 |
| **TACCL** | T1 "同构主机复制品"模型,**主机间 fabric 不可见**(internode 只有一个标量链路类):dgx_rail 的 spine 靠 sketch 人工禁跨轨绕开;T2 sketch 敏感:cube-mesh 机内必须人工收窄 2-relay 否则 routing MIP 零可行解——**每换一种布线都要人工重设计 sketch**;T3 大 chunk 画像撞硬编码时间哨兵/小数系数坑(需时间单位重标定 + α 取整才可解);T4 MIP 时限内留 gap(600s,gap 1.86%) | 异构/不规则图(ndv2_16 不能表达);多级 fabric(taper 不可见,§6.15 类 pod 超额场景无法感知);任何没有现成 sketch 的新布线 | CAP §2.2–2.3;`explore/2026-0710-experiment-summary-v4.md` §4;`script/common.sh make_padded_taccl_topo` 注释 |
| **SyCCL** | T1 分层同速模型(每层单一 link_spec):**异构混布不能表达**(400/200/100G 的 ndv2_16 直接出局);T2 机内部分网(`host_links:"nvlink"`)因上游 LP 解提取 bug(`DfsRateAlloc assert`)**实现不可用**,只能 nvswitch 全交叉近似(ndv2_32 成绩"可执行非忠实");T3 交换机 one-big-switch 理想化,端口级争用不可见(clos_32 的 224 个中继是真实额外负载);T4 spine 层实践上单脊(pod 分组 assert) | 机内 cube-mesh/环等部分网;异构速率混布;多脊 fabric | CAP §2.4("主机内部分网"行,两个上游 bug 的确诊:越界写已修、DfsRateAlloc 未修);`explore/2026-0710-experiment-summary-v4.md` §2.3 |

**汇总:谁适合什么**(诚实版,含对手的适用区):

| 方案 | 适合 | 不适合 |
|---|---|---|
| TE-CCL | 小规模(≤32)、静态均匀、**不规则/异构拓扑**(独占区) | 非均匀、动态、64+、多级图上要用对求解模式 |
| TACCL | 有现成 sketch 的同构复制拓扑、机内异构 NVLink 画像(1x/2x 混布是其看家本领) | 主机间 fabric 有结构的一切(rail taper/pod/多 spine)、无 sketch 新布线 |
| SyCCL | 规则分层 fabric(rail/pod 布线)、求解最快(40s) | 异构、机内部分网、多脊;one-big-switch 使中继代价被低估 |
| FAST | 强 scale-up 两层 fabric + 均匀/温和 skew(W 项,§1.3) | 弱 scale-up、fabric 有结构、执行期有不确定性 |
| MOSAIC | 两层 scale-up/scale-out 家族(rail/pod/双平面,§6.15/6.16)内的非均匀、动态、运行时不确定场景 | 强 scale-up + uniform/persrc(输 FAST 2–8%,§6.9);**无机内层的直连图**(regular_32/ndv2 类:corridor 前提不成立,那是 TE-CCL/包级喷洒的领地,`explore/2026-0710-experiment-summary-v4.md` §2.4);类数无闭式 prior 的布线(§8.2) |

### 1.5 定位:四个没人同时占住的空白

对照 12 家 related work(附录 B 全表):**(A)** 预计算调度的复用/缩放、
**(B)** 快速方法里的 fabric 级拓扑感知、**(C)** 执行期把释放容量转给
长尾的机制(所有静态 schedule 都默认执行 = 回放)、**(D)** 可执行的
fabric 内多路径分流。MOSAIC 的 prior 对准 B、wave feedback 对准 C,
corridor 结构为 A/D 留了接口(§8)。FAST 恰是 MOSAIC 在
"corridor 混合退化为全 NV 管、无 feedback"下的特例——一条干净的消融轴,
且现在是可运行的代码。

---

## 2. Related work(一段话版;12 家能力对照全表见附录 B)

离线 synthesizer(TE-CCL/TACCL/SyCCL/TACOS/HeteCCL)有 topology 级质量但
秒–小时级,且被 uniform 假设锁死(§1.2 实测;弱点全表 §1.4);ForestColl 明确排除 MoE
A2A;MCF-A2A 只解 equal-demand。在线方法里 **FAST** 是最强竞品(§1.3);
Themis 只均衡维度带宽失配,不碰数据 skew。**TCCL**(ASPLOS'24,
`tccl/`)证明"离线实测路径库 + 初始化期装配 + LD_PRELOAD 透明部署"在
真实系统可行——它是 MOSAIC 真实系统形态的工程参照(§8.3),但其路径
静态、不做 A2A、不见 skew。

---

## 3. 设计需求链(每个机制由哪条约束逼出来)

设计不是定理优先:每个机制先有逼出它的约束或实测失败,后有形式化。
模型:N 个 GPU 分成 H 台 server(每台 G=8);每个 GPU 的入口/出口各有
两条管——NIC 管(β_N=180G)与 scale-up 管(β_V=nv)。输入 demand matrix
D 在 collective 发起时才可知;目标 min makespan。

- **R1 时间预算**:流量几百 ms 一变 vs solver 秒–分钟级 ⇒ **LP/MIP 出局**,
  只许 pruning / greedy / closed-form(这也是"为什么没有用 DP"的答案的
  一半,另一半见 §4.2)。
- **R2 决策集必须 O(1)**:运行时要按反馈调"流量去向",候选去向必须少到
  每波每 dst 一次 O(N) greedy 切分能处理 ⇒ corridor pruning(§4.1)。
- **R3 feedback 学习交学费**:初值误差按 ~1/W 计入完成时间(实测 uniform
  初值开着 feedback 仍贵 5–17%)⇒ 必须有零成本的好初值 ⇒ 闭式
  waterfilling prior(§4.2)。
- **R4 共享干扰只有跑起来才可测** + **R5 prior 会过时**(MoE 漂移)⇒
  wave feedback(§4.4);三个实测病态(固化/振荡/积压污染)⇒ 三补丁。
- 形式化结论的真实角色,如实标定:pruning 论证 = **completeness check**
  (先手选、后验证);waterfilling 的"隔离最优"= **feedback 的职责边界**;
  "1/W regret"直觉界的假设(容量恒定、测量精确)在本系统不成立,它只是
  **设计指南**(解释 W 与 prior 质量的权衡),不是性能保证。

---

## 4. 算法与实现(含全部代码位置)

实现分两层:**runtime 管"发什么"**(`scheduler/mosaic_schedule.h`,
`SmartMosaicRuntime : SmartDeliveryObserver`,纯回调驱动、无定时器),
**node 层管"怎么发"**(复用 fast/teccl 的 min-epoch path follower,
`src/scheduler.h:25` 把 `mosaic_schedule` 映射到
`SmartFastScheduleScheduler`)。

### 4.1 Pruning:从 O(G²) 条 relay 链剪到 4 条 corridor

**候选空间**:每对 (s,d) 允许"发送侧 ≤1 relay + 接收侧 ≤1 relay"的链
(≤3 跳;更深的链被 α+重组成本论证排除,也与 FAST 链同深度保持公平)。
该类内每对仍有 O(G²) 条。

**剪枝规则**(设计期完成,两条):
1. **签名等价合并**:给链记 resource signature σ(p) = 占用的资源类型集
   (源 NIC↑ / spine / 目的 NIC↓ / 发送侧 NV / 接收侧 NV / 第三 rail)。
   同签名的链只差 relay 编号——运行时用轮转编号做负载均衡即可,合并为
   一类;
2. **支配剪枝**:同起终点下,σ(p₁) ⊆ σ(p₂) ⇒ p₂ 被 dominate,剪掉。

rail 布线下剪完恰余 4 类(**这一步在纸面完成,代码固化其结果**——运行时
没有、也不需要枚举-再-剪):

| corridor | 链 | 入目的 GPU 走哪条管 | 代码(`mosaic_schedule.h`)|
|---|---|---|---|
| DIRECT | s→d | NIC 管(跨 rail 经 spine) | `corridor_hops()`:298 |
| TX | s→(h,r_d)→d | NIC 管(发送侧 NV 换轨,避 spine) | 同上 |
| RX | s→(n,r_s)→d | NV 管(入口按源 rail 摊到 8 块 NIC) | 同上 |
| SPREAD3 | s→(h,j)→(n,j)→d,j 轮转 | NV 管(同 rail pair 唯一摊开手段) | 同上(轮转 j 的选择也在此) |

每对的可用 mask(同机 = 固定 NV 单跳不进混合;同 rail 跨机 =
{DIRECT, SPREAD lanes};跨 rail = 全部):`pair_corridors()`:290。
形式化角色 = completeness check:回答"为什么恰好 4 类、会不会漏"。

**v5 新增旋钮 `-mosaic_corridors C`(1..6,嵌套)**:1={DIRECT},
2=+RX,3=+TX,4=+SPREAD3(v4 默认),**5/6 = +额外 SPREAD lane**——
与 SPREAD3 同形状、同 pair 取不同过境 rail j(`corridor_hops()` 的
lane 语义),**只加 λ 粒度、不加新资源签名**。消融(§6.10,16 组)的
回答:"为什么是 4"= 4 是签名完备的最小集(C=1 逐位退化为 ECMP,1→2
一跳就是 6 倍差距);"数字有没有影响"= 完备之后 lane 加密还能稳定再拿
0.6–2.4%,故 v5 默认 C=6(§6.9)。

### 4.2 Greedy / closed-form:waterfilling prior(以及为什么不需要 DP)

把目的 GPU d 的入口看成两条并行水管:NIC 管(DIRECT/TX 喂)与 NV 管
(RX/SPREAD3 喂);机内流量 same_d 只能走 NV 管,跨机流量 cross_d 可分。
divisible load 下 makespan 最优 = 两管同时结束,**一维方程,闭式解**:

    x* = clamp( (β_V·cross_d − β_N·same_d) / (β_V + β_N), [0, cross_d] )

λ₀ 按 (NIC 管份额, NV 管份额) 设置,管内 2:1 细分。实现:
`init_lambda_from_demand()`(`mosaic_schedule.h:120`),每 dst O(N)、
全局 O(N²)、微秒级。stale-prior 实验用 `set_prior_demand()`:94,朴素
heuristic 初值用 `set_rx_init()`:103。

**为什么没有 DP**:问题结构给了闭式解——每 dst 的分配是"两条容量已知
的并行管上的可分负载",最优解就是等完成时间的 waterfilling,一步算出;
没有需要记忆化的子问题结构。corridor 类数变多时,K-管 greedy
waterfilling 已落码(`waterfill_pipes()`,等完成时间注水;X4 的三管
prior 是首个用例,§6.17),DP 仍为备选——见 §8.2。

**隔离最优性质**(Prop.):若 d 的两管不与他人共享,该分配使 d 的完成
时间恰等于 cut lower bound B1(d)。价值 = 划定 feedback 的职责边界:
prior 拿满"单 dst 能算对的部分",剩余误差只来自共享 fabric 干扰与测量
噪声,即 feedback 要修的全部。

### 4.3 Greedy:wave 装配(每波怎么切)

每个目的 GPU 一个独立控制环(`DstState`,λ[4] + wave 游标 + 逐波
`WaveStats`)。`emit_wave(d)`(`mosaic_schedule.h:363`)每波:
1. 逐 pair 配额 `quota = ceil(remaining / waves_left)`(末波清零);
2. 取该 pair 的 corridor mask(§4.1);
3. 用 **eff_λ = λ − stale_backlog/wave_total** 切分(anti-windup,
   `:383-405`:扣掉两波以前还没交付的字节;在途一波是正常流水);
4. 1KB 取整、余数给最大份额 corridor,逐份 `emit_chain()`:327。

### 4.4 在线 waterfilling feedback(与三个补丁)

波交付过 threshold(75%,判定在 `chunk_delivered()`:214)即测每条
corridor 的 effective throughput(交付字节 ÷ 占用时长),更新
λ ← (1−γ)λ + γ·(throughput 份额)(`update_lambda()`:468),然后立刻
放下一波(重叠掩盖波间空档)。"按实测 throughput 比例分配"是
divisible-load waterfilling 的不动点,所以更新方向是对的。

三个补丁,全部 bug 倒逼(时间顺序在先、控制论命名在后):

| 实测病态 | 补丁 | 代码 |
|---|---|---|
| 固化:慢 corridor 恰因慢而阈值前零交付,"保留旧信念"让热 NIC 每波被塞 19% 字节拖 4.3ms 尾 | 配了字节却零交付 ⇒ 份额按 0 记(**censoring bias** 修正)| `mosaic_schedule.h:501` |
| 振荡:小配额显快→加码→塞爆→压死→循环(λ 0.02↔0.26)| 单波增长封顶 ×1.5(**trust region**)| `:510` |
| 积压污染:超配残留让后两波测量连坐误判 | 分配扣 stale backlog(**anti-windup**)| `:394-405` |

另有 2% exploration floor 保证每条 corridor 持续可测。

### 4.5 与 htsim 的执行接口(通信语义级依赖如何保真)

- **注入** = `graph.nodes[src]->add_scheduled_chunk(...)`(两节点显式
  path;`inject_hop()`:347)。epoch 编码 `wave*4 + hop_index`:min-epoch
  选包下,**落后 GPU 的早 wave chunk 自动抢占共享端口**——straggler
  优先是编码顺带送的。
- **hop 链依赖**:hop k+1 只在 hop k 的 chunk 在 relay 完整交付后
  (`chunk_delivered()`:184 回调里 `next_hop++` 再注入)才发——正是
  CLAUDE.md 的"step B 等 step A 产生 chunk"铁律;relay 是 chunk 粒度
  store-forward,与 teccl/taccl/fast 映射同语义。
- **计量**:`count_delivery` 只在最后一跳为 true ⇒ validation 的
  `delivered_bytes` 恰等于 demand 总量,中间 relay 字节不掺水。
- **完成判定**:wave 自适应生成 ⇒ 总 chunk 数事前未知;runtime 自己数
  app 字节(`complete()`),`src/main_regular.cpp:817` 先跑到 complete
  再回填 expected。
- **刻意没有的东西**:无全局 barrier(N 个 dst 环只经共享链路耦合)、
  无墙钟对齐、无跨 dst 协调(§7 开放项)、无超过一波的预发(threshold
  每波只触发一次,在途 ≤2 波——§6.7 里 implicit flow control 的来源)。

**规模与开销**(n32, W=6;实测 `mosaic_report`):C=4 时 chains 18.6k、
hop 注入 35.9k;C=6 时 27.8k / 63.6k;prior O(N²) 一次 + 每波每 dst
O(N) 切分 + O(C) λ 更新,全部在事件回调里增量完成,无独立求解阶段。

### 4.6 FAST 移植与公平性旋钮(`scheduler/fast_schedule.h`,673 行)

algorithm 层逐行移植官方库:sender balance(`balance_server_pair()`:383,
对照 `FAST/simulation/scheduler/local.cpp`)→ server 级矩阵 Birkhoff
分解(`decompose()`:526,匈牙利 augmenting path `augment()`:585,对照
`FAST/simulation/algorithm/all2all.cpp`)→ 逐 stage `pull_row()`:464 +
receiver dispatch(`emit_chain()`:482)。执行映射复用 teccl path-step
runtime。公平性旋钮:官方 α-β guard(`guard_skip_balance()`:439,
`-fast_alpha_beta_guard`)、官方 scaling 路径的字节级等价
(`set_scaling_unit()`:230,`-fast_scaling`)、packet 调度语义对照
(`round_robin_ref()`:65,`-fast_rr_select`)。

### 4.7 Topology 与 workload 工具链

- **flex_rail**(`topo/flex_pod_graph.h:33`):dgx_rail 的参数化推广,
  旋钮 = hosts × gpus/host × nv_gbps × nic_gbps × taper × spines ×
  fabric_buffer_mb;nv=2400 时 ECMP 21829.2µs 与 dgx_rail 逐位一致。
- **demand 生成**(`tools/gen_nonuniform_a2a.py:32`):三种模式——
  hotdst(全局 Zipf 排名;注意其每 dst 机内:跨机比恒定的构造性质)、
  persrc(每 src 独立排名,`:49`,per-dst 结构互异,drift 实验必须用它)、
  khot(K 个等热 dst,`:38`);每 src 总量恒等于 uniform 契约,种子化。
- **pad-to-max fill**(`tools/fill_schedule_nonuniform.py`):四种 .sched
  格式的截断填充(teccl hop `:82` / taccl `:165` / syccl `:204`,链分配
  `assign_chains()`:62)。
- **降速旋钮**(v5,`topo/flex_pod_graph.h`):`-flex_degrade_gpu G
  -flex_degrade_factor F` 把单 GPU 的 NIC 双向降为 nic/F,调度器一律
  不知情、执行侧 pacing 按真实速率(§6.12 用)。
- **SyCCL converter 修复**(v5,`tools/syccl_generate_schedule.py`):
  SyCCL 在弱 NV 解里会把一个 pair 拆成多个 chunk(addr = dst + k·N);
  旧 converter 只按基础 chunk 记 demand,导致此类解在 htsim 里
  delivered=0 直接 FAILED。修复 = demand 按最终 chunk 表逐条生成
  (每 chunk 一行、携带各自字节数)。这是**替 baseline 修 bug**
  (§5.1 第 6 条),13 组 SyCCL@90 的成绩在修复后计入。

### 4.8 设计迭代记录(瓶颈驱动;每一版的依据)

| 版 | 改动 | 依据的实测瓶颈 | nv=2400 | nv=180 |
|---|---|---|---|---|
| v0 | per-dst-**server** λ + 硬波 barrier(W8) | — | 9660 | — |
| v1 | 控制粒度降到 per-dst-**GPU**;新增 SPREAD3;波重叠(75% 阈值) | server 级聚合速率掩盖热 GPU 的 direct 慢(λ_direct 被冷 GPU 钉在 0.31);同 rail 热流量无摊开手段;波尾排空管道 | 5090 | 12900 |
| v2 | censoring 修正(§4.4 补丁 1) | 慢 corridor 阈值前零交付 → 固化,热 NIC 拖 4.3ms 尾 | 3750 | 12800 |
| v3 | anti-windup + trust region(补丁 2/3) | λ 振荡 0.02↔0.26;backlog 连坐污染两波测量 | 3762 | 12970 |
| v4 | waterfilling prior(§4.2) | 中段 wave-0 均匀探索浪费;最优要 26% 走 NIC 管,从 uniform 学起太慢 | 3488(W8) | 12574(W8) |
| v4 定版 | W=8→6 | prior 好时波越少边界损耗越小 | 3557 | 12997 |
| **v5** | **SPREAD lane 加密:C=4→6**(算法零改动,只扩 λ 维度) | 16 组消融:C=6 在全部 5 个测点 ≥ C=4——SPREAD 轮转的负载均衡粒度不够,λ 分不开两条 lane 的份额 | **3526** | **12673**(nv90:16368→**16277**) |

方法论教训(可复用):(1) 反馈信号的 censoring bias 是致命的——被压死的
corridor 永远"无数据",必须把"配了没到"当最强负面证据;(2) 速率控制
不够,要**队列控制**(扣 backlog 比调 γ 有效);(3) 控制粒度必须匹配
异构性(热/冷 GPU 的最优配比天差地别,per-server 聚合直接抹掉信号);
(4) **消融要扫出设计空间边界**——"4 条"是完备性的答案,不是最优性的
答案(§6.9/6.10)。

### 4.9 数学分析(v5 新增;每条注明假设边界)

模型同 §3;ρ = β_V/β_N;us(B,β) = 8B/(1000β) µs。

**Prop 1(隔离最优)**:若 dst d 的两管不与他人共享,§4.2 的闭式分配使
d 的完成时间 = B1(d) = max(same_d/β_V, (same_d+cross_d)/(β_V+β_N))。
*边界*:共享 fabric 时不成立——那是 feedback 的职责(§4.2 已述)。

**Prop 2(反馈不动点 = waterfilling)**:更新 λ←(1−γ)λ+γ·(r_c/Σr) 的
不动点满足 λ_c ∝ r_c(corridor 有效带宽)。divisible load 下按可持续
速率比例分配恰使各 corridor 同时排空(等完成时间),即 K-管 waterfilling
的最优性条件;平稳速率下该映射是 (1−γ) 压缩,trust region 与 floor 只在
过渡期起作用(clip 后仍在单纯形内),不改不动点。*边界*:速率非平稳
(共享 fabric、他人的 λ 也在动)时无全局收敛保证——由 §6 实验回答。

**Prop 3(exploration floor 的保险费上界)**:C 条 corridor、floor ε,
每波至多 (C−1)ε 份额被强制放在非最优 corridor;最坏情形(这些字节零
贡献)完成时间膨胀 ≤ 1/(1−(C−1)ε)。C=4、ε=0.02 ⇒ ≤+6.4%;C=6 ⇒
≤+11.1%。实测保险费(§6.3:极弱 nv 且 prior 精确时 −1~5%)不超上界;
C=6 的实测净效应为正(§6.9),说明 lane 收益 > 增量保险费。

**Prop 4(corridor 盲的结构性代价 ⇒ crossover 定律)**:FAST 下,d 的
机内入流与所有**非 d 轨 landing** 的跨机字节(receiver dispatch)都要过
d 的单条 NV 下行管;例外只有 landing 恰为 d 的字节(sender balance 把
每个 server 对的字节摊平到 G 行 ⇒ 该份额 ≈ 1/G;`fast_schedule.h
emit_chain` 的 `landing_gpu != dst_gpu` 分支)。得**结构模型**

    T_FAST ≈ (same_d + (1−1/G)·cross_d) / β_V,

对实测的误差 ≤1.6%(见下);它 ≈ (1−σ')(1+1/ρ)·B1(d)(B1 由两管项
binding 时,σ'≈0.10)。MOSAIC 实测 ≤ (1+δ)·B1,δ≈0.07–0.15(§6.1)。
故弱侧优势 ≈ (1−σ')(1+1/ρ)/(1+δ):

| ρ | 结构比 1+1/ρ | 模型预测 (1−σ')(1+1/ρ)/(1+δ) | 实测 M/F(C=4) |
|---|---|---|---|
| 2:1 | 1.50 | 1.23 | 1.18 |
| 1:1 | 2.00 | 1.64 | 1.56 |
| 0.5:1 | 3.00 | 2.45 | 2.47 |

**FAST 实测贴死该结构模型**:代入本矩阵(hot dst=26,same=114.3MB,
cross=395.4MB,1−1/G=0.875)得 40875µs@nv90 / 10219µs@nv360,实测
40479 / 10126(偏差 −1.0% / −0.9%,来自 balance 取整与 guard 跳过的
3 对使 landing-on-d 份额略高于 1/G)——**dispatch 管就是它的 binding
约束,塌陷是结构性的而非实现噪声**。据此解析预测 crossover:解
(1−σ')(1+1/ρ*) = 1+δ(σ'=0.10, δ=0.10)得 ρ* ≈ 4.0,实测 ≈3.5:1
——同量级、同机理。*边界*:hot-dst 型矩阵(B1(hot) binding);
uniform 负载下 B1 项极小、模型退化(FAST 免付 dispatch 税的同轨份额
更大),对应 §6.11 的 FAST 主场。

**波数与 hop 开销**:初值误差 ~1/W 计入完成时间(设计指南,非保证,
§3 已标定);TX/RX 每字节至多 +1 次 NV 管占用、SPREAD +2,全部不占
spine——开销只加在该 regime 下未打满的管上,且 feedback 只有在实测
吞吐支持时才把份额挪过去(16 组 hop_bytes 列:C=4→6 hop 字节
+1.4~5.6%,端到端仍正收益)。

---

## 5. 实验设置

- **Topology**:flex_rail,扫 nv ∈ {2400,720,600,480,420,360,180,90}
  (NV:NIC 13:1 → 0.5:1),NIC 恒 180G、taper 恒 180G(§6.7/6.11 单独扫
  taper);host 形态 4×8 为主,8×4 / 2×16 见 §6.13。
- **NV:NIC 扫描点的真实硬件对照(问题:什么机器是 x:1)**——公开规格
  换算、量级口径;scale-up 取单向 per-GPU 聚合(双向规格 ÷2),GB/s×8:

| 机型/形态 | scale-up(单向) | 每 GPU NIC | NV:NIC | 对应扫描点 |
|---|---|---|---|---|
| DGX A100(NVLink3 + HDR 200G) | ~2400G | 200G | ~12:1 | 2400(13.3:1)|
| DGX H100(NVLink4 + NDR 400G) | ~3600G | 400G | ~9:1 | 720–2400 段 |
| H800(NVLink 削减版 + 400G) | ~1600G | 400G | ~4:1 | 720 ≈ crossover 上沿 |
| GB200 NVL72(NVLink5) | ~7200G | 400–800G | 9–18:1 | 强侧 |
| PCIe Gen5 盒(H100-PCIe/L40S + 400G) | ~512G(共享交换) | 400G | ≤1.3:1 | 180(1:1)|
| PCIe Gen4 推理机(L40S/4090 + 400G) | ~256G | 400G | ~0.6:1 | 90(0.5:1)|

  趋势:NIC 每代 200→400→800G,而大量商用推理/微调盘子是 PCIe-only
  (≤1:1);MoE 推理恰是 skew 最重的负载 ⇒ 弱侧 regime 真实存在且在
  扩大;crossover ≈3.5:1(C=4)意味着 H800 级机型已贴近边界。
- **Workload**:hot-dst Zipf(α∈{0.5,1.0,1.5,2.0}×seed∈{42,43,44})+
  persrc + khot family;总量恒 = uniform 契约(每 src 62MB);uniform
  组照常(§6.6/6.11)。
- **Baseline**:static ECMP、basic(此图全对全直连,退化 ≈ECMP,如实
  报告)、FAST(带官方 guard);**v5 新增弱侧离线 solver 全对比**
  (§6.8:schedule 生成与执行同一张图同参数;求解 >30min 或 infeasible
  记 SKIPPED 并给原因)。
- **执行底座公平性**:全方案共用 SchedulerCommon 端口事件/pacing/队列
  反压;`-ecn_threshold_bytes 0`(默认)语义 = **不覆盖队列内建默认**
  (`src/wait_queue.h:33` `set_congestion_thresholds`),即所有方案默认
  同一套 ECN/pause 底座——极端 backlog 运行里出现的 ECN 计数对全员一致。
- **Lower bound**(纯 cut,无 LP):每 GPU 两管入口 waterfilling cut ∨
  每 server G-NIC 入口 cut ∨ 每 GPU 两管出口 cut,取 max
  (`tools/x1_bounds.py`,与历史 `cache/nonuniform/x1_bounds.txt` 校验
  一致 ≤0.05%;支持 degrade / host 形态变体)。
- 单次 生成+模拟 ≤15 min;MOSAIC 参数全场统一(W=6, γ=0.7, floor=0.02,
  thr=0.75;**corridors:v4 组 C=4、v5 验证组 C=6,逐表标注**)。

### 5.1 Baseline 公平性协议(参数选择如何保证不偏袒)

1. **同一执行底座**:六方案共用 `scheduler/common.h`(SchedulerCommon)
   的端口事件/pacing/队列反压;ECN/pause 用同一套内建默认
   (`set_congestion_thresholds(0,0)` = 不覆盖,`src/wait_queue.h:33`);
   没有任何 per-方案网络参数。
2. **同一 workload 契约**:均匀 A2A 每对 2MB、包 1KB、store_forward;
   非均匀每 src 总量逐字节等于均匀契约
   (`tools/gen_nonuniform_a2a.py`,种子化)。
3. **solve/execute 拓扑一致铁律**:每个 nv 点 solver 用与 htsim 执行
   完全一致的 topology 描述**重解**(13 组:TE-CCL topo json 的
   `nv_gbps` 字段、SyCCL `nvlink.bw_mbpus`、TACCL intra beta 重标定);
   不存在"用 2400 的解跑 90 的图"。
4. **FAST 按官方最强形态跑**:所有对比默认带官方 α-β guard
   (`-fast_alpha_beta_guard`);另做 guard / round-robin 选包
   (`-fast_rr_select`)/ 官方 scaling(`-fast_scaling`)三个敏感性检查
   (03_fairness_fast):弱侧塌陷与这些选项无关,且默认 min-epoch
   mapping 对 FAST 更有利(rr 在 2400 反而劣化到 4894)。
5. **选包语义按 solver 分治、按对各家有利的方向选**:TE-CCL/TACCL/
   FAST/MOSAIC 用 min-epoch,SyCCL 用 round-robin——实验确定的各自
   最优映射(`teccl_schedule.h:185-191` 注释:换纯轮转让 TACCL
   dgx_rail +59%,故保留对它有利的语义)。
6. **pad-to-max 的 fill 语义做到 solver 最优**:padding 槽位不上网、
   src 只发实际字节、dst 收满即完成
   (`tools/fill_schedule_nonuniform.py`);TACCL 撞时间哨兵/小数系数时
   我们替它做时间单位重标定 + α 取整(`script/common.sh
   make_padded_taccl_topo`)而不是记 infeasible 零分;SyCCL converter 的
   chunk 拆分 bug(§4.7)也是修好后计入成绩(13 组 nv90)。
7. **solver 求解预算一致**:TACCL routing/scheduler 各 600s(与
   dgx_rail 基线相同),TE-CCL/SyCCL 跑到收敛;>30min 才记 SKIPPED
   (本轮 13 组无一 SKIP)。
8. **MOSAIC 无逐点调参**:全场一组默认;08/17 组证明参数带宽 ≤6–10%
   且默认在多个点非最优(从严口径);v5 的 C=6 是在全部 8 个 nv 点 +
   16 类负载/场景联合验证后才改的默认(§6.9),不是挑点。
9. **信息对称**:19 组降速实验中没有任何方案被告知降速(FAST plan 与
   MOSAIC prior 都按额定值);执行侧 pacing 按真实速率——硬件事实,
   对所有方案一致。
10. **下界核对**:所有表附 cut lower bound(`tools/x1_bounds.py`),
    贴界程度是第一 sanity check;对手数字异常先查映射(如 clos_32
    TE-CCL 的 min-epoch 异常已在 `explore/2026-0710-experiment-summary-v4.md` §2.4 归因为
    映射代价而非 solver 缺陷,如实标注)。

## 6. 评估

### 6.0 二十五组实验索引(`experiments/mosaic_x1/`)

| # | 实验 | 一句话结论 |
|---|---|---|
| 01_crossover | 主曲线(8 个 NV:NIC 点 × 方案 × bound,C=4) | crossover ≈3.5:1;弱侧 1.05–2.47× |
| 02_ablation | prior × feedback × corridor 混合 | feedback 三分法;单一摊开 corridor 崩 2.6× |
| 03_fairness_fast | guard / round-robin / scaling | FAST 弱侧塌陷与 mapping 无关 |
| 04_robustness | seed × α × family | 弱侧 2.1–2.65× 全稳;强侧 ±10% 矩阵依赖 |
| 05_drift | stale prior(persrc) | adapt 把 ±8% 压到 ≤3% |
| 06_uniform | uniform A2A sanity | nv=90 连 uniform 都赢 1.17×;2400 慢 9% |
| 07_scale_n64 | 64 GPU | crossover 成立;wave = implicit flow control |
| 08_param_sensitivity | W / γ / threshold | 全带宽 ~10%,无逐点调参 |
| 09_multihot | K∈{2,4,8} 等热 dst | 全胜(90 点 2.1–2.5×) |
| 10_msg_size | 0.25× / 4× 契约 | 幅度稳定(90 点 2.38×/2.49×) |
| 11_taper | spine taper {90,180,360}G | FAST 恒 10126(fabric 盲);MOSAIC 自适应全最优 |
| 12_lambda_trace | 热 GPU λ 轨迹 | 2400→RX 重、90→DIRECT/TX 重(waterfilling 两端) |
| **13_full_baseline** | 全 baseline 非均匀 ×{13.3,2,0.5}:1 | 拓扑感知 LP 也躲不开 pad-to-max(输 ≥1.37×) |
| **14_uniform_congestion** | uniform × taper × ECN | 无拥塞逼平;taper 只罚 ECMP;FAST uniform 免疫(如实) |
| **15_skew_sweep** | α∈{0.5..2.0} × 全方案 | α=2 强侧也反超 FAST 1.18×;@90 距界 +0.4% |
| **16_corridor_count** | C∈{1..6} 消融 | C=1≡ECMP;签名完备>数量;C=6 全测点 ≥C=4 |
| **17_threshold** | thr∈{0.3..0.95} | 全带宽 ≤6%,0.75 非精调 |
| **18_synthesis_speed** | 合成时间 vs 规模(32–256 GPU) | µs(MOSAIC)/ms(FAST)/几十秒–十几分钟(solver) |
| **19_degraded_nic** | 未通告 NIC 降速 | 热点降速把 13:1 从平手变 1.44–1.50× 赢 |
| **20_host_shape** | 8×4 / 2×16 形态 | crossover 恒成立;2×16 全 nv 赢(含强侧 1.09×) |
| **21_design_iter** | {C, thr} 联合验证 | **C=6:主线 8 点全 ≥FAST(1.00–2.49×),v5 默认** |
| **22_x2_pod** | X2:pod 布线+4 spine(corridor 族再生成) | 全 9 点赢 FAST 1.19–2.59×,多点贴界;prior 公式零改动 |
| **23_x3_rail2p** | X3:双平面 rail + 平面不对称 | **32 组已重跑**:九点全 ≥ FAST(1.02–2.22×);奇偶盲区(结构性)成立,原 0.73× 属已删除的旧 corridor 族 |
| **24_x4_threepipe** | X4:三管 corridor 族 + K-管注水 prior | 平面拆分族让 2400+B/4 达 1.02×(本组数字可复现;"旧族"对照列失效,幅度见 §6.17 更正);known prior 在弱 NV 值 22% |
| **25_x2full_spine** | X2-full:显式 vs 隐式 spine 钉法 | 规则:G%S==0 隐式、互质显式;S 不改变 pod 结论 |

### 6.1 主结果 v4:crossover 曲线(C=4;hot-dst zipf1.0/seed42,µs)

| NV:NIC | nv | bound | ECMP | FAST | **MOSAIC** | M/F | 距 bound |
|---|---|---|---|---|---|---|---|
| 13.3:1 | 2400 | 3327 | 21829 | 3541 | 3557 | 1.00× | +6.9% |
| 4:1 | 720 | 4532 | 21829 | 5068 | 5111 | 0.99× | +12.8% |
| 3.3:1 | 600 | 5229 | 21829 | 5988 | **5728** | **1.05×** | +9.5% |
| 2.7:1 | 480 | 6181 | 21829 | 7369 | **6657** | **1.11×** | +7.7% |
| 2.3:1 | 420 | 6798 | 21829 | 8746 | **7679** | **1.14×** | +13.0% |
| 2:1 | 360 | 7556 | 21829 | 10126 | **8566** | **1.18×** | +13.4% |
| 1:1 | 180 | 11330 | 21829 | 20244 | **12997** | **1.56×** | +14.7% |
| 0.5:1 | 90 | 15105 | 21829 | 40479 | **16368** | **2.47×** | +8.4% |

- FAST 的塌陷超线性:2:1 已落后 bound 34%,0.5:1 落后 2.68×、比 ECMP
  还慢 1.85×——"全走 NV 管"的结构性代价(W7)。
- **p99 tail**(2400/360/90):ECMP 20107(恒);FAST 3046/7169/29334;
  MOSAIC 3026/**6013**/**12595**(弱侧 tail 好 2.3×)。
- 调度开销同为在线量级:MOSAIC 微秒级增量,FAST plan 亚 ms–ms 级
  (32 GPU 中位 0.64ms,规模化到 109ms@256,§6.14)。

### 6.2 公平性(FAST 是不是被跑弱的?——不是)

总协议见 **§5.1**(十条,覆盖全部 baseline);本节只放 FAST 专项敏感性
检查(03 组)的数字:

| 检查 | 结果 |
|---|---|
| 官方 guard | 2400 触发 3 对(3596→3541);**nv≤720 触发 0 对**(它只防 α 主导,不防弱 β_V) |
| round-robin 选包(`-fast_rr_select`) | nv≤360 **逐位不变**;2400 反而劣化到 4894——默认 min-epoch 是对它有利的 mapping |
| 官方 scaling(`-fast_scaling`) | 2400 最优档 −0.3%;90 三档全部 40478.6——瓶颈在 dispatch 带宽不在 stage 数 |

### 6.3 Ablation(feedback 值多少;hot-dst seed42,µs,C=4)

| 变体 | 2400 | 720 | 360 | 180 | 90 |
|---|---|---|---|---|---|
| **prior + feedback(v4 默认)** | **3557** | **5111** | **8566** | 12997 | 16368 |
| W=1 纯 prior | 3972 | 5918 | 8816 | **12331** | 16117 |
| prior + static | 3872 | 5192 | 8660 | 12136 | 16037 |
| uniform 初值 + feedback | 3743 | — | — | 13434 | 19123 |
| uniform 初值 + static | 9449 | — | — | 14698 | 29365 |
| 朴素 RX heuristic(static) | 3740 | 5889 | 10517 | 21024 | 42040 |
| RX heuristic + feedback | 3598 | — | 9304 | — | 27091 |

**三分法**:(i) 强/中 nv:feedback 净赚 3–14%(修 prior 看不到的共享
干扰);(ii) prior 错误时的保险:uniform+static 崩 2.7×/1.8×,feedback
救回到 5–17% 内;朴素单 corridor 崩 2.57× ⇒ 混合比例本身必要;
(iii) 极弱 nv 且 prior 精确:中性偏负(−1~5%,exploration 的保险费)。

### 6.4 Robustness(seed × α × family;FAST/MOSAIC,µs)

| workload | 2400 | 360 | 90 |
|---|---|---|---|
| seed43 | 4245/4230(1.00×) | 10163/**8633**(1.18×) | 40626/**15346**(2.65×) |
| seed44 | 3386/3687(0.92×) | 10152/**8424**(1.21×) | 40582/**15517**(2.62×) |
| α=0.5 | 2697/2683(1.01×) | 4169/**3783**(1.10×) | 16647/**7516**(2.21×) |
| α=1.5 | 5132/**4918**(1.04×) | 18043/**14375**(1.26×) | 72145/**28198**(2.56×) |
| persrc | **2537**/2850(0.89×) | 3178/**2808**(1.13×) | 9585/**6196**(1.55×) |

弱侧全稳(2.2–2.65×);α=1.5@90 距 bound 仅 +4.7% 而 FAST 比 ECMP 慢
2.19×。**强侧如实呈现 ±10% 矩阵依赖**:温和 skew 下 FAST 的全局
permutation 协调赢 8–11%——这是 regime 图,不是单点碾压(C=6 后收窄到
2–8%,§6.9;α 扩展到 2.0 后强侧反转,§6.10)。

### 6.5 Drift(stale prior;persrc,先验按 seed43 算、跑 seed42)

| nv | fresh+static | stale+static | fresh+adapt | stale+adapt |
|---|---|---|---|---|
| 2400 | 3077 | 3125 | 2850 | **2815** |
| 360 | 3080 | 2950 | 2808 | **2726** |
| 90 | 6273 | 6774(+8%) | 6196 | 6326(+2%) |

feedback 让性能对 prior 过时不敏感(≤3% vs static 的 ±8%)。方法学注记:
hotdst family 做不了此实验(每 dst 机内:跨机比恒定 ⇒ prior 天然不随
seed/α 变,§4.7),drift 结论以 persrc 为准。

### 6.6 Uniform / 规模 / 多热点 / 尺寸

- **Uniform A2A**(06):2400:FAST 2345 / MOSAIC 2576(−9%,主场认输);
  360:2622/2671(−1.8%,逼平);90:5229/**4485**(+1.17×,弱
  scale-up 连 uniform 都赢)。细化(taper/ECN)见 §6.11。
- **64 GPU**(07):1GB fabric buffer 下 ECMP(全部)与 FAST(nv90)被
  1.78GB 热点 backlog 冲爆 FAILED,MOSAIC 全 PASSED(wave 限 in-flight =
  **implicit flow control**);4GB buffer 公平版:0.97× / 1.15× /
  **2.39×**(2400/360/90),crossover 形状在 2× 规模成立。
- **多热点**(09,K∈{2,4,8}):全胜——360:1.18/1.24/1.27×;90:
  2.47/2.12/2.10×。
- **尺寸**(10,0.25×/4× 契约):1.18×/2.38×(512KB)与 1.18×/2.49×
  (8MB,4GB buffer 公平版);幅度对尺寸不敏感。

### 6.7 taper 与参数敏感性

- **taper**(11,{90,180,360}G@nv360,非均匀):FAST 恒 10126(从不用
  spine,W1 实测演示);ECMP 剧烈敏感(43558/21829/17845);MOSAIC 全
  最优且近乎平坦(8475/8566/8896)——taper 变小时 feedback 自动压
  DIRECT 份额,**fabric 旋钮的适应零配置**。C=6 复验:8212/8418/8675
  (`x21_v1_robust.log`),仍全档最优。
- **参数**(08,@nv360):W∈{1..12}/γ∈{0.3..0.9}/thr∈{0.5,0.75,0.9}
  全变化带宽 ~10%(7991–8816);全文一组默认打全场,默认在该点甚至
  非最优(W=2 更好)——从严口径。threshold 三个 nv 点细扫见 §6.10。

### 6.8 全 baseline 非均匀对比(13 组;zipf1.0/seed42,µs)

问题 1 的正面回答。schedule 生成与执行同一张 flex_rail 图(§5.1 第 3
条):TE-CCL topo json 带 `nv_gbps`;SyCCL config 的 nvlink 层带宽同步
(2400G=0.3 → 90G=0.01125 MB/µs);TACCL intra beta ×(2400/nv) 重标定
再做 padded 时间单位重标定。pad-to-max + fill 契约与 §1.2 相同;
本轮**无一 SKIP**(TACCL@90 也在 600s 预算内解出):

| nv | bound | ECMP | TE-CCL | TACCL | SyCCL | FAST | **MOSAIC(C4)** | M vs 最强 solver |
|---|---|---|---|---|---|---|---|---|
| 2400 | 3327 | 21829 | 17450 | 17451 | 15927 | 3541 | **3557** | 4.48×(SyCCL)|
| 360 | 7556 | 21829 | 11762 | 15302 | 15717 | 10126 | **8566** | **1.37×**(TE-CCL)|
| 90 | 15105 | 21829 | 22640 | 31299 | 23231 | 40479 | **16368** | **1.38×**(TE-CCL)|

- **拓扑感知救不了 pad-to-max**:TE-CCL 的 LP 明知 nv=360/90——它把
  RX 型中继解了出来,从 2400 的 17450 改善到 11762/22640——但 padded
  均匀计划的结构税仍让它输 MOSAIC 1.37×/1.38×,且每个 nv 点重解一次
  分钟级(§6.14)。
- SyCCL@90 自己启用了 chunk 拆分(每 pair 2 chunk)仍 23231(输 1.42×);
  TACCL@360/90 = 15302/31299(输 1.79×/1.91×)。
- **弱侧排序反转的含义**:@90 时 FAST(40479)从最强在线变成全场最差
  (比 ECMP 慢 1.85×),而三家 solver 聚在 22.6–31.3k;**MOSAIC 是唯一
  贴 bound 的方案**(+8.4%)。"在线"不是分水岭,**结构假设 + 执行期
  调整才是**。
- 数据:`13_full_baseline/results.csv`,原始日志
  `cache/nonuniform/x13_solver_batch.log`。

### 6.9 设计迭代:C=6 新默认(21 组;主线 zipf1.0/seed42,µs)

16 组消融(§6.10)发现 lane 加密一致有益后,对 {C, thr} 组合做全 nv 点
+ 多负载联合验证(`21_design_iter/results.csv`):

| nv | bound | FAST | C4(v4) | **C6(v5)** | **C6/FAST** | C6t95 | C5t95 |
|---|---|---|---|---|---|---|---|
| 2400 | 3327 | 3541 | 3557 | **3526** | **1.004×** | 3487 | 3487 |
| 720 | 4532 | 5068 | 5111 | **4960** | **1.022×** | 5151 | 5215 |
| 600 | 5229 | 5988 | 5728 | **5610** | **1.067×** | 5775 | 5846 |
| 480 | 6181 | 7369 | 6657 | **6590** | **1.118×** | 6786 | 6747 |
| 420 | 6798 | 8746 | 7679 | **7542** | **1.160×** | 7528 | 7517 |
| 360 | 7556 | 10126 | 8566 | **8418** | **1.203×** | 8306 | 8274 |
| 180 | 11330 | 20244 | 12997 | **12673** | **1.597×** | 12003 | 12017 |
| 90 | 15105 | 40479 | 16368 | **16277** | **2.487×** | 15834 | 15850 |

- **C=6(thr 保持 0.75)在全部 8 点严格优于 C=4,且 vs FAST 全点
  ≥1.00×**——v4 唯一输的 720 点(0.99×)翻正(1.022×)。改动是单参数,
  非逐点调参(§5.1 第 8 条)。
- thr=0.95 在两端更好(180:12003)但中段略差、且在 720 重新输 FAST
  (5151 > 5068)——**不采纳**:"主线全点不输"优先于局部再快 5%。
- **C=6 十六项稳健性复验**(`x21_v1_robust.log` + 16 组,vs FAST,
  2400/360/90):seed43 1.00×/1.19×/2.61×;seed44 0.98×/1.18×/2.60×;
  α0.5 1.02×/1.11×/2.19×;α1.5 0.98×/1.27×/2.61×;α2.0
  1.13×/1.30×/2.64×;persrc 0.95×/1.22×/1.47×;uniform
  0.92×/0.95×/1.16×;khot@90 2.46×/2.16×/2.10×;taper@360 全档最优;
  热点降速@2400×4:8892(vs FAST 13301 = **1.50×**,比 C4 的 9253 再进
  一步);host 形态 8×4:1.03×/0.99×/2.18×,2×16:**1.24×**/1.32×/1.78×。
- 如实:C6 并非处处 ≥C4(zipf1.5@2400 5215 vs 4918;2×16@90 24380 vs
  23523)——**"主线 8 点全不输 FAST"是 C=6 的稳定性质,"处处优于
  C4"不是**;强侧非主线负载仍有 2–8% 矩阵依赖(§7 开放项 1)。

### 6.10 skew 扫描(15 组)与 corridor/threshold 消融(16/17 组)

**skew**(hot-dst zipf,seed42;`15_skew_sweep/results.csv`):

| α | 2400:FAST/MOSAIC(C4) | 90:FAST/MOSAIC(C4) | @90 距 bound |
|---|---|---|---|
| 0.5 | 2697/2683(1.01×) | 16647/**7516**(2.21×) | +21% |
| 1.0 | 3541/3557(1.00×) | 40479/**16368**(2.47×) | +8.4% |
| 1.5 | 5132/**4918**(1.04×) | 72145/**28198**(2.56×) | +4.7% |
| 2.0 | 7711/**6542**(**1.18×**) | 100459/**37658**(2.67×) | **+0.4%** |

- **skew 越重优势越大,且把 crossover 推向强侧**:α=2.0 时 13.3:1 上也
  赢 1.18×(nv=360:25119/19359 = 1.30×)。机理 = Prop 4 的 B1(hot)
  随 α 增大而更 binding。
- **@90/α2.0 距 cut bound 仅 +0.4%**:v4 的"bound 紧度未知"开放项在
  重 skew 处闭环——不是 bound 松,是计划真的近似最优。
- ECMP 随 α 恶化(12675→43421)且 nv 无关(单 NIC 直发墙)。
- **solver 行(α 逐点重解 pad,@2400,全部 PASSED)**——pad 税随 skew
  超线性增长:

| α | pad | TE-CCL | TACCL | SyCCL | MOSAIC(C4) | M vs 最强 solver |
|---|---|---|---|---|---|---|
| 0.5 | 8MB | 7187 | **4915** | 6779 | **2683** | 1.83×(TACCL)|
| 1.0 | 32MB | 17450 | 17451 | 15927 | **3557** | 4.48×(SyCCL)|
| 1.5 | 64MB | 31091 | 28076 | 27685 | **4918** | 5.63×(SyCCL)|
| 2.0 | 64MB | 43271 | 38917 | 38147 | **6542** | 5.83×(SyCCL)|

  α≥1.5 时三家 solver 退化到 ECMP 量级(32998/43421)——pad-to-max 在
  极端 skew 下把"有计划"的价值全部吃掉;且每个 α 都是一次分钟级重解
  (TACCL ~685s、TE-CCL 125–130s、SyCCL ~50s,
  `x13_solver_batch.log` RUN15 段时间戳)。
- **corridor 数**(16 组,`16_corridor_count/`):C=1 逐位 ≡ ECMP
  (21829.2,sanity);1→2(+RX)是最大跳变(@2400 → 3674,5.9×);
  签名完备(≤4)贡献数倍,lane 加密(4→6)再拿 0.6–2.4%;hop 字节随
  C 单调增(@2400 1.57×→1.77× app 字节),端到端仍正收益。**"为什么
  是 4":4 = 资源签名完备的最小集;"数字有没有影响":有,但完备之后
  是 1–2% 量级,完备之前是数倍量级。**
- **threshold**(17 组,`17_threshold/`):thr∈{0.3..0.95} × 3 nv 全
  带宽 ≤6%,无悬崖;0.75 三点均非最优但都在最优 5.5% 内——**0.75 不是
  精调产物,是"哪里都不差"的中位选择**;两端失效假设(太低=测量噪声、
  太高=流水线断)在此范围内未出现,因为"每波只触发一次 + 在途 ≤2 波"
  的结构性重叠保护。

### 6.11 uniform × 拥塞 × ECN(14 组;问题 2)

| nv | taper | ECMP | FAST | MOSAIC(C4) |
|---|---|---|---|---|
| 2400 | 180 | 7852 | **2345** | 2576 |
| 2400 | 90 / 60 | 15605 / 23533 | **2345 / 2345** | 2715 / 3164 |
| 360 | 180 | 7852 | **2622** | 2671 |
| 360 | 90 / 60 | 15605 / 23533 | **2622 / 2622** | 3030 / 2864 |
| 360 | 90+ECN | 16567(3360 cuts) | **2622**(0 cuts) | 3206(4793 cuts) |

- **无拥塞:逼平成立**——MOSAIC 距 FAST −1.8%(360)/−9%(2400),
  全员距 cut 界(B2=2133,与 taper 无关,因跨轨流量可经 NVLink 换轨)
  +10~21%;C=6 复验 2553/2766/4502(@90 赢 1.16×)。
- **taper 拥塞只罚路径不可变的 ECMP**(贴其结构界 7466/14933/22400
  +4~5%):FAST 的 i→i 在 uniform 下根本不过 spine,**天然免疫——这是
  对 FAST 有利的场景,如实呈现**;MOSAIC 的 DIRECT 探索份额付 3~23%
  后仍 5~7× 好于 ECMP。**"拥塞下反超 FAST"的战场在非均匀负载**
  (§6.7:FAST 恒 10126,MOSAIC 全 taper 档最优)。
- ECN 显式收紧不改排序(全方案同一设置;§5.1 第 1 条)。

### 6.12 未通告的 NIC 降速(19 组;问题 8 的核心新证据)

单 GPU 的 NIC 双向降为 nic/F,**没有任何方案被告知**(§5.1 第 9 条);
热 = 最热 dst 26,冷 = gpu 7;基线(无降速)FAST 3541/10126、MOSAIC
3557/8566:

| nv | 降速点 × 倍数 | ECMP | FAST | MOSAIC(C4) | M/F |
|---|---|---|---|---|---|
| 2400 | 热 ×2 | 35066 | 6864 | **4669** | **1.47×** |
| 2400 | 热 ×4 | 70070 | 13301 | **9253**(C6:**8892**)| **1.44×(1.50×)** |
| 2400 | 冷 ×2 / ×4 | 21829 | 4703 / 9356 | **4540 / 9007** | 1.04× |
| 360 | 热 ×2 / ×4 | 35066 / 70070 | **10126 / 13301** | 10194 / 13759 | 0.97–0.99× |
| 360 | 冷 ×2 / ×4 | 21829 | 10126 | **8567 / 8335** | 1.18–1.21× |

- **执行期不确定性改写 regime 图**:基线平手的 13.3:1 点,热点 NIC 被
  干扰后 MOSAIC 赢 1.44–1.50×——W5(静态置换无人回收)的直接实测。
  MoE 场景里这类干扰(邻租户、降速链路、PCIe 争用)是常态而非异常。
- 如实:360+热点降速 FAST 略胜 1–3%——其 ingress 本经 G 块 NIC 摊开
  (i→i + dispatch),对单 NIC 降速有结构免疫;nv=360 时 MOSAIC 的 NV
  替代容量有限。两家都远离 cut 界(10069),单 NIC 降速的界本身偏松
  (§7 开放项 5)。
- 冷 GPU 降速:MOSAIC@360 反而比无降速略好(8335 < 8566)——feedback
  把降速 NIC 的份额挪走顺带减轻争用;ECMP 恒 21829(它从未用过那块
  NIC 的富余)。

### 6.13 host 形态(20 组;问题 9——corridor 剪枝的泛化)

同一张 32-GPU demand 矩阵(pair 结构与 host 划分无关),三种形态:

| 形态 | nv=2400 | nv=360 | nv=90 |
|---|---|---|---|
| 4h×8g(主线,C4)| 3541/3557(1.00×)| 10126/**8566**(1.18×)| 40479/**16368**(2.47×)|
| 8h×4g:FAST/MOSAIC(C4) | 7901/7942(0.99×) | 9076/**8976**(1.01×) | 34690/**16113**(**2.15×**)|
| 2h×16g:FAST/MOSAIC(C4) | 2341/**2147**(**1.09×**)| 10842/**8487**(**1.28×**)| 43342/**23523**(**1.84×**)|

- **crossover 形状在两个极端形态都成立,corridor 集合零改动**——§4.1
  剪枝按资源签名而非具体 G 值定义,G=4(SPREAD 轨少)与 G=16(机内
  主导)自动适配。
- **G=16 时强侧也赢 1.09×**:host 只剩 2 台,FAST 的 server 级 Birkhoff
  只有一个非平凡置换,全局协调优势消失;MOSAIC 的 per-dst 混合不依赖
  server 数。C6 复验:2×16@2400 = 1886(1.24×)。
- 如实:8h×4g@90 FAST(34690)比 ECMP(27831)还慢(W7 在小 G 下更痛);
  2h×16g@90 ECMP(21572)略好于 MOSAIC(23523,距界 21773 仅 +8.0%,
  但 ECMP 恰好贴界)——机内绝对主导时"什么都不做"躲开全部中继开销。

### 6.14 schedule 合成速度(18 组;问题 6/11——faster 支柱)

口径:MOSAIC = prior 构造(`mosaic_plan_us`,含 λ 初值全量计算);
FAST = 完整 plan(balance+Birkhoff+dispatch,`fast_plan plan_us`);
solver = 求解 + 转换 wall time。非均匀 zipf1.0/seed42,3 次取中位
(`cache/nonuniform/x18_plan_times.log`):

| N GPU | MOSAIC | FAST | TE-CCL | TACCL | SyCCL |
|---|---|---|---|---|---|
| 32 | **20µs** | 636µs | 147s | ~840s | 40.5s |
| 64 | **72µs** | 3.0ms | **>1800s 超时†** | 跳过† | 262s |
| 128 | **195µs** | 12.7ms | — | — | — |
| 256 | **~0.8ms** | **~105ms** | — | — | — |

† 本轮实测(`cache/nonuniform/x18_solver_n64.log`):TE-CCL 8host×8gpu
2MB uniform 在 30min 墙钟内无解(rc=124,LP 变量数随节点数爆炸);
SyCCL 262s(2× 规模 6.5× 耗时);TACCL 的 sketch 需要为 8 host 人工
重设计,未尝试(32-GPU 已 ~840s)。另注:同规模下**换一个 pad 尺寸就
要重解**(RUN15 实测:α=0.5 的 8MB pad TE-CCL 重解 130s、SyCCL 50s、
TACCL 686s,`x13_solver_batch.log` 时间戳)。

- **三个量级台阶**:µs–亚 ms(闭式 prior,O(N²):20→72→195→816µs,
  与 N² 吻合)→ ms(Birkhoff ~O(S⁴):0.64→3.0→12.7→~105ms,S=服务器
  数)→ 几十秒–十几分钟(LP/MIP)。**256 GPU 处差距 ~130×**:FAST 的
  105ms 已接近 MoE 流量漂移的时间尺度,而 MOSAIC 仍在 1ms 内。
- 对 MoE 的含义:矩阵每几百 ms 一变 ⇒ 只有 µs–ms 级能"每个 collective
  现装配";solver 即使离线摊销,pad 尺寸随矩阵变(§6.10:α=1.0 pad
  32MB、α=2.0 pad 64MB),**每个新矩阵形状都要重解**。MOSAIC 的 λ 还可
  跨 collective warm-start(§8.1),连 prior 重算都能省。

### 6.15 X2:pod 布线 + 多 spine——corridor 集按拓扑再生成(22 组)

原 §8.2 的 X2 计划,提前完成。**flex_pod**(`build_flex_pod_graph`):
每 host 的 8 块 NIC 接同一台 leaf、4 条 spine、路由钉死 `(s+d)%4`。
pod 下 TX(换轨)无意义,corridor 族按 §4.1 的剪枝流程**重新推导**:
`DIRECT + RX-lane m`(落到 dst 机第 m 个 sibling 再 NV 转发;**落点同时
决定入口 NIC 与 spine**——一个杠杆摊两个瓶颈,`WIRING_POD`);
**两管 waterfilling prior 公式零改动复用**。MOSAIC C=5,zipf1.0/seed42:

| nv | leaf 上行(超额) | bound | ECMP | FAST | **MOSAIC** | M/F | 距界 |
|---|---|---|---|---|---|---|---|
| 2400 | 1440(1:1) | 3327 | 17506 | 6820 | **3719** | 1.83× | +11.8% |
| 2400 | 720(2:1) | 6654 | 18250 | 13397 | **6788** | 1.97× | **+2.0%** |
| 2400 | 360(4:1) | 13309 | 20347 | 26533 | **13430** | 1.98× | **+0.9%** |
| 360 | 1440 | 7552 | 17506 | 10126 | **8487** | 1.19× | +12.4% |
| 360 | 720 | 7552 | 18250 | 14629 | **8348** | 1.75× | +10.5% |
| 360 | 360 | 13309 | 20347 | 27686 | **13827** | 2.00× | +3.9% |
| 90 | 1440/720/360 | 15104 | 17.5–20.3k | 40479(恒) | **15.6–16.7k** | 2.4–2.6× | +3.3~10.5% |

- **X2 论点成立**:corridor 族按拓扑再生成、prior 公式一字不改——§7
  旧局限 3("rail 布线专用")的方法论半边就此关闭。附加组((360,720)):
  C∈{1..6} = 17939/10218/8388/**8279**/8348/8529(C=1≈ECMP sanity;
  平台在 C=4/5);uniform_init +4.2%、static +3.3%(结构贡献主体);
  seed43 1.93×、zipf1.5 1.62×(稳健)。
- **超过原预期**:§8.2 预期 vs ECMP hash 撞车 1.2–1.5×,实测 vs ECMP
  1.12–4.71×、vs FAST 1.19–2.59×,且 2400 系列距割界 +0.9%~+2%。
- **FAST 在 pod 上全面恶化**:共享 leaf 折断其 i→i 摊开(无超额时 6820
  vs rail 同点 3541);4:1 超额时 26.5–27.7k **比 ECMP 还慢**(W1 fabric
  盲);nv=90 恒 40479。
- 数据:`22_x2_pod/`(41 行,含消融/稳健组);复现命令见其 README。

### 6.16 X3:双平面 rail——relay 即平面选择(23 组)

原 §8.2 的 X3,按"视 X2 结论决定"执行。**flex_rail2p**
(`build_flex_rail2p_graph`):每 GPU 两块 90G NIC 接两套独立 rail fabric
(总带宽与 X1 逐字节相同),跨机路由钉死平面 `(s+d)%2`;
`-flex_planeb_factor F` 把 B 面全 fabric 降为 1/F(未通告的平面不对称)。
MOSAIC 调度器零改动上阵。主表(C6/FAST,µs;全表与 C4 行见
`23_x3_rail2p/`):

> **⚠ 2026-07-28 更正(32 组重跑,详见 23 组 README)**:下表 MOSAIC
> 列产生于 X4(24 组)**原地替换掉**的旧 RAIL2P corridor 族,未留旧族
> 开关 ⇒ 在任何已提交版本上都无法复现(v5 与当前提交逐位一致地给出
> 重跑表;ECMP/FAST 逐位不变 ⇒ 拓扑/demand 未变)。**以重跑表为准**,
> 尤其 **"2400+B/4 输 FAST 0.73×"已作废**。

旧族表(历史,不可复现):

| nv | 对称 | B/2 | B/4 |
|---|---|---|---|
| 2400 | 7168/**6842**(0.95×) | **6422**/6842(1.07×) | 9385/**6842**(**0.73×**) |
| 360 | **8584**/10126(1.18×) | **9635**/10126(1.05×) | 10433/**10126**(0.97×) |
| 90 | **18362**/40479(2.20×) | **21219**/40479(1.91×) | **33854**/40479(1.20×) |

重跑表(当前代码;各点取 C4/C5/C6 最好 vs FAST,全表见 23 组 README):

| nv | 对称 | B/2 | B/4 |
|---|---|---|---|
| 2400 | **6349**/6842(**1.08×**) | **6418**/6842(1.07×) | **6679**/6842(**1.02×**) |
| 360 | **8862**/10126(1.14×) | **9466**/10126(1.07×) | **9581**/10126(**1.06×**) |
| 90 | **18209**/40479(2.22×) | **22894**(C4)/40479(1.77×) | **29491**(C4)/40479(1.37×) |

(ECMP:25040/50264/100710,逐位同旧归档。)**重跑后九点全部 ≥ FAST
(1.02×–2.22×)**;新增现象:弱 NV+重不对称处 C4 反优于 C6,强 NV 处
C6 远优于 C4——"C 随 regime 移动"再添例证。

- **奇偶盲区(本组最有价值的结构性发现)**:`(s+d)%2` 钉死下,一切
  i→i 同轨中继跳的 GPU id 和恒为偶数 ⇒ **永远钉在 A 面**。FAST 因此三档
  逐位不变——对 B 面降速"意外免疫",代价是对称时也只用一半 fabric
  (6842 = X1 同点 3541 的 1.93×);MOSAIC 的 TX/RX/SPREAD 中继类同样
  被钉 A 面,**"relay 即平面选择"只对跨轨 DIRECT(奇偶混合)成立**。
- 结果:MOSAIC 靠 DIRECT 奇偶混合 + feedback 在 360/90 全档赢
  (1.05–2.20×),B/2 时甚至优于自身对称(坏平面信号加速 λ 收敛);
  ~~但 **2400+B/4 输 FAST 0.73×**~~——**该结论已作废(32 组重跑)**:
  旧族在此点确实缺能进 B 面的中继类(奇数和 inter hop,签名含 B 面
  spine),但当前族在该点是 1.02×,不再输 FAST。
- **这反而强化 X2 论点**:corridor 集必须按拓扑生成,且剪枝的签名分析
  能精确指出缺哪一类;**该缺口由 X4 的三管族补上(§6.17;修复幅度以
  当前代码为准,"9385→6717"不成立)**。下界注记:
  聚合 NIC 割在 B/4 处偏松(平面结构未编码),见 23 组 README。

### 6.17 X4 / X2-full:corridor 族补全与显式 spine 钉法(24/25 组;§8.2 收尾)

**X4 三管族修复奇偶盲区(24 组)**:把 RAIL2P 的 DIRECT 按 pair 平面拆成
DIRECT_A/DIRECT_B 两条 corridor,dst 入口变成真三管可控(NV/NIC-A/NIC-B);
prior 用通用 **K-管 greedy waterfilling**(`waterfill_pipes()`,等完成时间
注水,§4.2 的承诺落码,K=3 用例)。结果(vs 23 组旧族 / FAST):

- ~~**X3 两个薄弱点全部修复**:2400+B/4 从 0.73× 到 1.02×(9385→6717)~~
  **32 组更正**:24 组自身数字**逐位可复现**(重跑 C5 九点 + known 五点
  全中),但被比较的"旧族"列来自已删除的代码(见 §6.16 更正框),
  "0.73×→1.02×"这一**幅度**不成立。成立的是**方向**:平面拆分族让
  2400+B/4 达到 1.02×;且当前族最优不是 X4 的 C=5 而是**同族 C=6**
  (2400/360 六点全面更好)。修复主要来自**结构**而非先验——强侧
  unknown ≈ known(6717 vs 6790)。
- **known prior 的价值随 feedback 变慢而增长**:90+B/4 处 known 快 22%
  (23876 vs 30552;vs FAST 1.70×)——弱 NV 下探索代价高,好初值省掉的
  学习波次更值钱("1/W 遗憾"指南的反向印证)。
- 如实:对称/弱 NV 处旧族仍略优(360 对称 8584 vs 8956)——**族选择也
  是 regime 相关的**,与 16 组"C 随 regime 移动"同构。
- 诚实注记:两层 GPU 系统 per-dst 入口物理管数 ≤ 端口数,K>3 的独立管
  尚未在实验族中出现;`waterfill_pipes()` 为异构多层 fabric 预留。

**X2-full 显式 per-spine 钉法(25 组)**:pod RX-lane 的 spine 绑定,
隐式(lane=落点序)vs 显式(lane≡spine,`-mosaic_explicit_spine`),
S∈{3,4,5}@(nv360, 2:1):

| S | 隐式 | 显式 | 判定 |
|---|---|---|---|
| 3(互质) | 8465 | **8393** | 显式 +0.9% |
| 4(整除) | **8348** | 8904 | 隐式 +6.7% |
| 5(互质) | 8720 | **8487** | 显式 +2.7% |

"S 与 G 互质时隐式覆盖有偏"的疑虑证实并闭环:**规则 = G%S==0 用隐式
(轮转天然均匀且保留落点自由度),否则用显式**——一行判断。附带发现:
**FAST 的 spine 指纹 = 2i mod S,S 为偶数时恒偶 ⇒ 只用一半 spine**
(S=4:14629;S=3/5:10126)——与 §6.16 奇偶盲区同根:同 index 中继 +
模运算钉死 = 系统性类缺失,这类结构盲点正是 corridor-按-拓扑-生成要
消灭的对象。MOSAIC 领先 FAST 1.19×(奇 S)–1.75×(偶 S)、ECMP
2.05–2.18×。数据:`24_x4_threepipe/`、`25_x2full_spine/`。

---

## 7. 讨论:局限与开放问题(v5 审阅后)

v4 的 11 项 reviewer 质疑对应 §6 各小节;v5 新增回答:全 baseline
(§6.8)、skew 边界(§6.10)、corridor/threshold 依据(§6.10)、拥塞
(§6.11)、执行期干扰(§6.12)、形态(§6.13)、合成速度(§6.14)。
**v4 开放项 1(bound 紧度)部分闭环**:α=2.0@90 距 cut +0.4%(§6.10),
证明 bound 族在重 skew 处是紧的;中段 +7~15% gap 的 oracle 标定仍值得做
但优先级降低。声明 out of scope:硬件现实性(机型标定/NIC 维度/proxy)。
### 7.1 MOSAIC 的缺点:什么情况下表现不好(M 项,与 W/S/T 对仗)

自我解剖与 §1.3(FAST 的 W 项)、§1.4(solver 的 S/T 项)同一标准,
全部有实测出处:

| # | 缺点 | 实测幅度 | 出处 |
|---|---|---|---|
| M1 | **强 scale-up + 全局协调友好负载输 FAST**:uniform/persrc/温和 skew 下,FAST 的全局 permutation(一对一、无 incast)是真优势,MOSAIC 的 per-dst 贪心没有跨 dst 协调 | uniform@2400/360:0.92×/0.95×;persrc@2400 0.95×;seed44@2400 与 zipf1.5@2400 均 0.98×(皆 C6) | §6.9/6.11,21 组 |
| M2 | **无机内层的直连图不适用**:corridor 剪枝前提是"scale-up + scale-out 两层";直连稀疏图上没有 NV 管可借 | 未跑(前提不成立即出局);该区由 TE-CCL/包级喷洒占据(basic@regular_32 = 3638 最快) | §1.4 总表;`explore/2026-0710-experiment-summary-v4.md` §2.4 |
| M3 | **exploration floor 是常付的保险费**:每波 (C−1)ε 份额强制喂非最优 corridor;prior 已精确时反馈中性偏负;拥塞 fabric 上探索流量还会挨 ECN | 极弱 nv:−1~5%(§6.3);uniform+taper:比 FAST 多付 3~23%、4793 次 ECN cut(§6.11);上界 +6.4%(C4)/+11.1%(C6)(Prop 3) | §6.3/6.11/§4.9 |
| M4 | **corridor 族是设计期产物**:环境超出族的表达力时 feedback 无处可逃——X3 旧族在 2400+B/4 输 FAST 0.73×(该族已被 X4 原地替换删除,当前族该点 1.02×,见 §6.16 更正框),直到 X4 补三管族才修复;且族选择本身 regime 相关(对称/弱 NV 处三管族反而比旧族慢 2~5%) | 0.73×(修复前);族间差 ±2~5% | §6.16/6.17 |
| M5 | **单 collective 内从坏初值恢复不完全**:W 波的学习预算有限,RX-init@90 只恢复到 27091(完整版 16368,差 1.66×) | 1.66× | §6.3;开放项 2 |
| M6 | **中继字节开销加在稀缺资源上**:hop 字节 = app 字节的 1.2–1.8×(随 nv 变);机内绝对主导的形态下"什么都不做"反而更好 | 2h×16g@90:ECMP 21572 < MOSAIC 23523(0.92×) | 16/20 组 |
| M7 | **无全局保证**:隔离最优(Prop 1)只在 per-dst 管不共享时成立;共享 fabric 下收敛性、最优性、跨 dst 公平性均无证明,只有实证(距界 +0.4%~+21% 随负载浮动) | gap 波动区间 +0.4%~+21% | §4.9 Prop 2 边界;§6.10 |

一句话画像:**MOSAIC 在"结构性瓶颈可被多 corridor 摊开、且环境有不确定
性"时最强;在"全局协调本身是最优解"(强 scale-up 均匀负载)、"没有
可摊开的第二资源"(直连图、机内独大)、"环境超出 corridor 族表达力"
(未建模的钉死维度)三种情况下表现不好。** 前两种是定位边界(§1.4
总表如实划给对手),第三种可用"按拓扑重推族"缓解但不能根除(X4 证明
可修,也证明需要修)。

### 7.2 仍开放的问题

1. **强 scale-up + 温和/中性 skew 的非主线负载**(= M1 的翻盘路径):C=6 后主线 8 点全
   不输,但 seed44/zipf1.5/persrc/uniform 在 13.3:1 仍输 FAST 2–8%
   (§6.9)——全局 permutation 协调是真优势;翻盘需要跨 dst 协调
   (波内配对 permutation coloring),超出本原型。
2. 单 collective 内从坏初值**恢复不完全**(RX-init@90 恢复到 27091 vs
   完整版 16368)→ 跨 collective 的 λ warm-start。
3. ~~corridor 集与 prior 公式是 rail 布线专用~~ **v5.1 已部分关闭**
   (§6.15/6.16):pod 与双平面 rail 下 corridor 族按同一剪枝流程再生成
   (pod:DIRECT+RX-lane;双平面:rail 族零改动兼任平面选择),两管
   prior 公式两处均一字不改。仍开放的部分:类数多到无闭式 prior 的布线
   (如异构多层 fabric)需要 K-管 greedy waterfilling / DP(§4.2)。
4. prior 用了 β_V:β_N 拓扑参数(FAST 不用)——辩护:uniform 初值 +
   feedback 完全不用该知识,在 nv=90 仍赢 2.1×(§6.3)。
5. **单 NIC 降速的 cut 界偏松**(§6.12:兄弟 NIC 不受影响,B2 不动)
   ——紧界需按 corridor 结构算 max-flow,留作 bound 工具下一步。
6. C 的最优值随 regime 移动(16 组:2400→5、360/90→3,差距 ≤2.5%)
   ——启动时用同一 waterfilling prior 对 C∈{4,5,6} 闭式预估再选,
   零运行时代价(§8.1)。

---

## 8. 未来工作(吸收自原研究计划,按优先级)

### 8.1 近期(不依赖新机制)
- **E0-b oracle**:TE-CCL 对真实非均匀矩阵直接解 LP,标定中段 bound
  gap(重 skew 处已由 §6.10 的 +0.4% 闭环);
- **跨 collective warm-start**:MoE 迭代间复用 λ(开放项 2),同时给出
  "漂移幅度 vs 收敛代价"的曲线;
- **C 的启动期自动选择**(开放项 6):prior 闭式预估 C∈{4,5,6},零
  运行时代价。

### 8.2 探索 topology(原计划 X2/X3——**v5.1 已完成**,见 §6.15/6.16)
- ~~X2 pod 布线 + 多 spine~~:**完成**(22 组)。实测超预期:vs FAST
  1.19–2.59×、多点距割界 ≤5%;"corridor 集按 topology 生成"从论证变实测。
- ~~X3 双平面 rail~~:**完成**(23 组)。零改动迁移在 360/90 全档赢;
  发现"奇偶盲区"(同轨中继恒钉 A 面),2400+重不对称需补 B 面中继类。
- ~~per-spine 显式 path-pinned corridor~~:**完成**(25 组):互质 S 用
  显式、整除 S 用隐式,一行规则。
- ~~K-管 greedy waterfilling~~:**已落码并有用例**(`waterfill_pipes()`,
  X4 三管 prior,24 组)。B 面中继类(奇数和 inter hop)经 X4 证明
  **不再必要**——三管拆分已修复全部薄弱点。
- 仍留:真正类数爆炸的异构多层 fabric(>3 独立入口管需要多 NIC 平面
  ≥3 的硬件形态)——工具已备,等一个真实拓扑。

### 8.3 真实系统落地(TCCL 给的三条工程路径,`tccl/` 源码摸底)
1. **实测干扰矩阵替代解析 prior**:TCCL 的 pathfinder 坚持实测(NUMA
   干扰只有测才看得见)且用组合法抗爆炸(机内路径拼跨机路径);MOSAIC
   的 corridor 两两干扰可离线实测成查找表,替代签名线性叠加假设。
2. **库→运行时交付格式**:TCCL 的 XML→channel 实例化
   (`tccl/src/tccl/tccl.cc:85-446`)+ `LD_PRELOAD` + `NCCL_ALGO`
   回退链,是 MOSAIC 真实形态的模板(`NCCL_ALGO=MOSAIC,FAST,RING`)。
3. **chunk→channel 晚绑定**:NCCL 本就把 collective 切成 chunk 轮转到
   多 channel;把走廊候选预实例化成 channel、把轮转换成按队列深度选,
   即 corridor 混合的真实实现(不动传输层)。

### 8.4 原 MOSAIC 愿景中未实现的机制(诚实清单,详见附录 A)
- **D1 走廊晚绑定**(逐 chunk 按端口队列深度选路)与 **D2 捐赠晋升**
  (完成事件把优先级槽转给长尾)——本原型用 per-dst wave feedback
  (≈原 D3)覆盖了大部分收益,D1/D2 是更细粒度的补充;
- **模板库 + 轨道压缩 + PCT 标定**的完整库愿景——本原型把"库"固化为
  4 条 corridor,大愿景在 X2/X3 的多类 corridor 下才需要。
- 理论问题(组合间隙上界、长尾回收竞争比、库规模–质量权衡、干扰矩阵
  可线性化条件)保留为第二支柱。

---

## 9. 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# MOSAIC(v5 默认 C=6)/ FAST @ NV:NIC = 2:1
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail -flex_nv_gbps 360 \
  -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -workload A2A -all_to_all -store_forward -quiet -demand_file $D
#   ... -scheduler fast_schedule -fast_alpha_beta_guard ...
# MOSAIC 旋钮: -mosaic_corridors(v5 默认 6;v4 组=4;消融 1..6)
#   -mosaic_waves(6) -mosaic_gamma(0.7) -mosaic_floor(0.02)
#   -mosaic_threshold(0.75) -mosaic_static -mosaic_uniform_init
#   -mosaic_rx_init -mosaic_prior_demand_file <stale.txt>
# FAST 公平性: -fast_alpha_beta_guard -fast_rr_select -fast_scaling <B>
# topology: -flex_nv_gbps -flex_taper_gbps -flex_hosts -flex_gpus_per_host
#   -flex_fabric_buffer_mb -flex_degrade_gpu <g> -flex_degrade_factor <f>
#   -topology flex_pod(-flex_spines S,X2)/ flex_rail2p(-flex_planeb_factor F,X3/X4)
# X2/X4 旋钮: -mosaic_explicit_spine(互质 S 用)-mosaic_known_planeb <F>(静态异构先验)
# workload: tools/gen_nonuniform_a2a.py --mode {hotdst,persrc,khot}
# cut 下界: python3 tools/x1_bounds.py --demand $D --nv-gbps 360 \
#   [--degrade-gpu G --degrade-factor F] [--gpus-per-host 4|16]
# 13 组 solver 侧描述与批日志: cache/nonuniform/teccl_flex*_topo.json,
#   syccl-grb/config/flex*-a2a-32mb.json(result json 在 build/result-flex*), cache/nonuniform/taccl_flex*_topo_*.json,
#   cache/nonuniform/x13_solver_batch.log
```

实验总集 `csg-htsim/experiments/mosaic_x1/`(21 目录,results.csv +
plot.png + README,画图 `plot_all.py`);原始日志
`csg-htsim/cache/nonuniform/x1_*.txt` 与 `x13/x15/x18/x21_*.log`;代码:
`scheduler/mosaic_schedule.h`、`scheduler/fast_schedule.h`、
`topo/flex_pod_graph.h`、`tools/{gen_nonuniform_a2a,
fill_schedule_nonuniform, syccl_generate_schedule, x1_bounds}.py`。
配套背景文档:`Note/explore/2026-0710-experiment-summary-v4.md`(四 solver × 五 topology 全景)、
`Note/explore/2026-0708-solver-topology-capabilities.md`(solver 能力边界 + FAST §7)、
`Note/paper_fast_summary.md`(FAST 论文精读)。

---

## 附录 A:与原 MOSAIC 愿景的映射(实现了什么、简化了什么)

原计划(已删除的 `idea_moe_subschedule_assembly.md` v3.1)设想三层:
离线**子调度模板库**(模式枚举 + 轨道压缩 + htsim 标定 (u, PCT))→
**装配期** LP 按签名防超卖 → **执行期** D1 走廊晚绑定 / D2 捐赠晋升 /
D3 波次重装配。本原型的对应与偏离:

| 愿景组件 | 本原型的实现 | 偏离原因 |
|---|---|---|
| 模板库(离线求解 + PCT 标定) | 固化为 4 类 corridor + 2 条可选 spread lane(设计期 pruning,§4.1),无离线求解、无 PCT 标定 | rail 布线下 4 类已 complete;PCT 被"直接实测 throughput"取代(feedback 就是在线标定) |
| 装配 LP(签名防超卖) | 闭式 waterfilling prior + greedy 切分(§4.2/4.3,零 LP) | 用户约束"禁 LP";两管结构给了闭式解 |
| D3 波次重装配 | wave feedback(§4.4)= D3 的轻量变体(不重跑装配,只调 λ) | 剥离+重装配的输入(完成计数)被 per-corridor throughput 取代 |
| D1 走廊晚绑定 | 未实现(corridor 在 emit 时绑定) | wave 粒度已覆盖大部分收益;逐 chunk 晚绑定留作 §8.4 |
| D2 捐赠晋升 | 未实现;min-epoch 的"早 wave 抢占"是弱化版(§4.5) | 同上 |
| P-* path-pinned 模板(空白 D) | 未实现;X2 载体(§8.2) | rail 单 spine 下 spine 不构成入口容量,打不出分 |
| E0 动机实验 | 完成(§1.2) | — |
| FAST 消融轴 | 完成且可运行(§4.6) | — |

## 附录 B:12 家 related work 能力对照(关键数字为本仓库实测或论文自述)

| 方案 | 解什么 | 离线/在线,耗时 | 非均匀 A2A | fabric 感知 | 执行期调整 | 保证 |
|---|---|---|---|---|---|---|
| TE-CCL | 任意图 LP | 离线;147s@32GPU(实测) | 可表达但慢;pad-to-max 实测输 MOSAIC 1.37–4.9×(§6.8) | 显式图+α,buffer≡0 | 无 | LP 最优 |
| TACCL | 复制主机 MIP+sketch | 离线;~14min@32(实测) | 否(padding 还有哨兵/小数坑) | fabric 不可见 | 无 | 无 |
| SyCCL | 分层 fabric 搜索 | 离线;40s@32(实测) | 否 | 层栈+taper,one-big-switch | 无 | 无 |
| TACOS | 时间展开图贪心 | 离线;1.08s@128NPU | 否 | switch=β×d | 无 | 无 |
| Themis | 维度顺序贪心 | 在线 | 否(带宽失配≠数据 skew) | 维度级 α-β | 每次重调度 | 无 |
| ForestColl | 树打包 | 离线;<1s@32 | **明确排除 MoE A2A** | switch 带宽 | 无 | 吞吐最优 |
| Blink | MWU 树打包 | 在线(作业启动) | 否 | 不建模 | chunk 级 MIAD | 单根最优 |
| MCF-A2A | 分解 MCF LP | 离线;秒级@数百 | equal-demand only | 无 switch | 无 | LP 精确 |
| AutoCCL | NCCL 参数调优 | 在线;几迭代 | 否 | 不建模 | 参数级 | 无 |
| **FAST** | alltoallv Birkhoff | 在线;0.6ms–12.7ms@32–128(实测) | **是** | 两层假设写死;fabric 粒度盲(实测 §6.7/6.11) | 无(静态置换;§6.12 实测代价) | 最坏 2.12× |
| HeteCCL | 异构 SMT+CEGIS | 离线;0.9s@16 | 否 | shared-group 容量 | 无 | 渐近带宽最优 |
| TCCL | PCIe 路径发现 | 离线实测 + 运行时零成本 | 否(不见 skew) | 不建模,全靠实测 | 无(路径静态) | 无 |
| **MOSAIC(本文)** | alltoallv corridor 混合 | 在线;20µs–0.8ms@32–256(实测) | **是** | **β_V:β_N + taper 自适应(实测)+ 运行时干扰自适应(§6.12)** | **wave feedback(实测)** | 隔离 waterfilling 达割界(Prop 1);floor 保险费有上界(Prop 3);全局无保证 |
| TCCL | PCIe 路径发现 | 离线实测 + 运行时零成本 | 否(不见 skew) | 不建模,全靠实测 | 无(路径静态) | 无 |
| **MOSAIC(本文)** | alltoallv corridor 混合 | 在线;20–195µs@32–128(实测) | **是** | **β_V:β_N + taper 自适应(实测)** | **wave feedback(实测,§6.12)** | 隔离 waterfilling 达割界;floor 保险费有上界(§4.9);全局无保证 |

---

## 10. v6(2026-08-01):26-38 组进展综述与统一方案定稿

v5 之后的 13 组实验把 MOSAIC 从"25 组时代的 C6+反馈"推进到**统一方案
MOSAIC-U**。本节只给主线与结论,逐组细节见各自报告(文件都在 `Note/`)。

### 10.1 机制探索简史(26-33 组)

- **26/27 组(D2 捐赠/skew 优先序,`explore/2026-0727-x26-27-donation-phase.md`)**:
  MOSAIC-P = skew_epoch2 + rewaterfill,每 nv 快 3-5%;E1 冷放大负对照、
  E4 份额捐赠被否(→ `fail.md`)。
- **28 组(wave-ramp,`explore/2026-0727-x28-wave-ramp.md`)**:MOSAIC-R = P + 冷源
  ramp 双门,全点支配 P;ramp-hot 与固定 wave size 证伪。
- **29-31 组(cube-mesh + 机内 LP,`explore/2026-0727-x29-flex-cube.md`、
  `explore/2026-0727-x30-mosaic-lp.md`、`explore/2026-0727-x31-mosaic-lp-v2.md`)**:
  异构机内画像下 corridor 由 LP 表驱动(离线产物,~26ms/拓扑);
  FAST 差距归因 = 落点集中,jdist 统一摊布。
- **32 组(双悖论+冷源限速,`explore/2026-0728-x32-paradox-pacing.md`)**:PACE
  (`-mosaic_cold_pace`,三自动门)取代 R 的大部分收益;HOSTL
  (per-host λ)定位为故障仪表;23 组重跑结案。
- **33 组(FAST 运行时同步,`explore/2026-0728-x33-fast-runtime-sync.md`)**:FAST
  基线的 stage 序修正(largest-first)与 async 语义确认——对手最有利
  引用值 = 3419@2400。
- **34 组(O1 配对着色,`explore/2026-0731-x34-pair-color.md`)**:PC =
  epoch×K + (d−s) mod K 的 Cayley 对角着色,把 M1(uniform 输 FAST)
  翻盘(uniform@360 1.047×),刷新 4 个 SOTA 点;PC×R 冲突暴露
  (E3 层抽空破坏 Latin square)。

### 10.2 超大规模(36 组,n=256→1024 weak scaling)

契约:per-src 总量恒定 65MB(base=2MB·31/(n−1)),hot-dst zipf1.0
seed42;flex_rail hosts=n/8×8,主线参数原样放大;LP 系 solver 按声明
排除(数百节点无法出解)。仿真器为本组做了 chunk pool list+per-epoch
桶索引优化(**全库逐位回归**,n128 提速 2.46×)。方法学细节与原始
日志:`explore/2026-0801-x36-scale-1024.md`、`36_scale/`。

**主表**(completion_us;FAILED = 默认 1GB fabric buffer 溢出丢包,
仅诊断;buffer 补偿 = 对手最有利地加大至 PASSED):

| n | LB | **MOSAIC C6+PC** | FAST | ECMP |
|---|---|---|---|---|
| 256 | 40571 | **41547(1.024×)** | 默认 FAILED(0.6M drops)→ +4GB 59410(1.46×) | FAILED → +4GB 145371(3.58×) |
| 512 | 74180 | **78977(1.065×)** | FAILED → **+16GB 107934**(1.46×) | FAILED → +4GB 287413(3.87×) |
| 1024 | 139139 | SKIPPED(仿真>20min 政策) | 同(构造 2.70-3.11s 已测) | FAILED(47.6M)→ +16GB 557815(4.01×) |

**per-A2A 构造时间 = 大规模隐性 overhead**(demand 每次变,必须重跑
的部分;不含 MOSAIC 一次性拓扑初始化):

| n | MOSAIC(waterfilling 先验) | FAST(stage/BvN 全构建) | 比值 | FAST 构造/其 collective |
|---|---|---|---|---|
| 32 | 71.6µs | 1.87ms | 26× | 0.18 |
| 256 | 1.30ms | **111.1ms** | 85× | **1.87×(构造已超 collective!)** |
| 512 | 4.35ms | **1.494s** | 343× | **13.8×** |
| 1024 | 20.3ms | **2.70-3.11s** | 133-153× | collective 未测(政策弃) |

读数:MOSAIC+PC 随规模**贴界**且默认 buffer 零丢包(4/16GB 下数字
逐位不变,双重中性证明);FAST/ECMP 的 blast 热点把默认 buffer 打爆;
FAST 构造复杂度从"启动开销"变成 MoE 主循环上的乘法税。

### 10.3 switch-centric(37 组,纯 clos + incast 等现象)

拓扑 = 两层纯 Clos(host 100G,单上行,degree-1 ⇒ 差异全在时间调度);
细节:`explore/2026-0801-x37-switch-centric.md`、`37_switch_centric/`。

**incast(m 个跨 leaf aggregator,per-src 预算守恒)**:completion
全员贴 ingress 割界(≤0.3%),**差异化在 buffer 轴**——

| 方案 | incast1 min-buffer | 模型 |
|---|---|---|
| MOSAIC W=6/12/24 | (256,512] / (128,256] / (64,128] MB | **C/W**(wave 定律,三连减半全命中;W 加倍 completion 代价 ≤0.11%) |
| FAST / ECMP | (1024,2048] MB(默认 1GB 打爆,53.8 万丢包) | ≈0.9C(blast 汇聚) |

即 25 组"wave = implicit flow control"的 incast 定量版:MOSAIC 的
buffer 足迹可用 W 线性调低,blast 方案只能加 buffer。

**其他现象**:oversubscription 扫(1:1→8:1)下均匀流量是 ECMP 主场
(全线 1.00×,MOSAIC+PC 跟到 1-3%),skew 流量反转(PC 四点精确贴界
40791.3、对 oversub 免疫,ECMP 漂到 +2.8%)——**MOSAIC+PC 是唯一
两边都 ≤3% 的**;spine 钉链 collision 是调度不可恢复的负边界(四方案
同贴 4.00× 钉链割,fail.md §2.12);PC 纯 incast 中性(单 sink 序
无关)。

### 10.4 统一方案 MOSAIC-U(38 组定稿)

**动机**:SOTA 表一度按 regime 特调(90 用 PC+R+PACE、360 用 PACE、
2400 用 R……)。38 组把机制收敛为一个方案:

```
MOSAIC-U = C(wiring 定)+ W(拓扑可调)
         + -mosaic_pair_color_auto    # diag-CoV ∧ rail2p-wiring ∧ lp_spread 三重自动门
         + -mosaic_cold_pace 1        # skew / 2×slack / taper<nic 三门
         [+ cube LP 表(离线产物,可选)]
```

R/HOSTL/E 系退役(`fail.md` §1)。35 cell 全域扫描
(`experiments/mosaic_x1/38_unified/`):**10 新 SOTA(含 rail2p 360
对称 −3.4%、64GPU −2.3%、drift −1.1%)、8 持平/逐位、10 代价点
(最大 cube 100x16 +4.8%;rail 2400 +3.0% 是唯一 vs FAST 落后点
0.975×)**。新发现并门控的机制冲突:PC×lp_spread(cube mix 归因
+3.1%,LP 产物优先)。

**解读**:统一的税集中在两类 regime——R 曾独占优势的"极端 NV 比"
(90/2400/khot 多热点)与 PACE 三门未覆盖的"弱且均匀机内画像"
(cube 100x16);其余 26/35 cell 统一方案 ≥ 历史特调。这个交换
(≤4.8% 最坏税 ↔ 零 regime 判断、全自动门)是部署语义的净收益。

### 10.5 当前完整状态

> **2026-08-05 增补**:41 组 = EPS 立足点审计与 MOSAIC-L 账本核,
> **已经晋升门转正为 §11 的 MOSAIC-U2——本节以下的 U 配置与数字
> 转为历史,现行 SOTA 见 §11**。

- 实验组:38(01-38 全归档,`experiments/mosaic_x1/`);
- 主报告:本文档;
- 常设表:`SOTA.md`(统一方案版);负结果:`fail.md`;
- 操作手册:`handoff.md`。

---

## 11. v7.2(2026-08-08):MOSAIC-U2b 定稿——账本核 + 双通道广播协议(现行 SOTA)

**先说人话**:本节是现在真正在用的方案,三句话讲完。第一句(x41):
**从"学"改成"记账"**——拥堵的大头是我们自己已经分配出去的数据,
这笔账可以精确记,不用靠小份试探去学;每个物理口(网卡上下行、机内
通道、交换机上行)记一本"已分配未送达"的账,发每个波次时把配额填
到"最忙的口最早忙完"的走廊组合里,等价于在线直接朝理论下界优化。
第二句(x42):**送达计数靠遥测传播,不假设零成本同步**——每台机器
把自己几个口的实测计数每 100µs 发布一次(有界陈旧,SSP 式)。
第三句(x44):**"每台机器自己重算全局分配"这条捷径被我们自己测
死了**——视图差一拍的机器会把 11-39% 的字节算到不同走廊,账越记越
歪;所以分配决策也走广播:每个目的地在发波次时把"这一波怎么分"
广播出去(每波 0.3-2.6KB),别人只存你最新一份;再加一个几微秒的
**决策错峰**(同一瞬间触发的机器排个队再决策,不然互相看不见)。
这套全可实现的协议比"假设重放成立"的旧数字整体只慢 **0.1%**,比
回滚到 λ 老方案快 **5%(39 胜 18 负)**。

- 探索记录:审计 `explore/2026-0805-x41a-eps-foundation-audit.md`、x41 裁决
  `explore/2026-0805-x41b-eps-ledger.md`、x42 同步模型
  `explore/2026-0805-x42-ledger-sync.md`、x44 协议重构
  `explore/2026-0808-x44-lg-bcast.md`;数据 `experiments/mosaic_x1/
  41_ledger/` + `bench_runs/u2bl.csv`(67 cell 正跑,全 PASSED、
  drops 0);新方案对比一律走基准闸(CLAUDE.md)。

### 11.0 晋升与修订记录(透明,含判据击穿)

- **x41 门**(26 成对 cell,零成本同步版):18 胜/7 负/1 平;预立
  −5% 判据被 3 cell 击穿,依聚合交换转正并如实记录。
- **x42 修订(应指出:x41 的数字建立在零成本精确全局同步上,不可
  作为优势 claim)**:实现 oracle/strong/snap/none 四种同步模型重测。
  定稿 = **snap-100µs**(非阻塞周期遥测)。全 67 cell 重跑后
  **vs λ 旧方案(回滚线):42 胜 / 13 负 / 3 平**;负 cell 全部落在
  同一 **NV 饱和族**(@90 各流量 + 机内主导形态,最深 m512k@90
  0.80,归因见 §11.5);**vs oracle 聚合税 +1.6%**。裁决:不回滚
  ——赢面(降速 2.3-2.4×、漂移逐位免疫、khot/n64/形态 +7-12%、
  2400 vs FAST 翻正)远大于税族,λ 核单开关可回。oracle 数字降级为
  参考(基线缓存中 `u2_oracle_ref`),CLAUDE.md 新增"禁零成本同步"
  铁律。
- **x44 修订(应指出:x42 的 charges 靠"确定性重放"免传,该假设
  被自家探针证伪)**:重放输入不止 demand 矩阵(账本快照/剩余量/波
  时机都是分布式状态);fork 探针实测,视图差一个 100µs 周期的 rank
  每波把 **11-39% 的字节**分到不同走廊,且 chg 累加不自愈。改为
  **通道 A(实测计数,100µs 周期)+ 通道 B(策略广播,每波)+
  决策错峰(3µs×8 槽)**,细分 fabric 账目声明其 fabric 前提(发端
  可选 spine;纯 ECMP 哈希 fabric 用聚合兜底档,价目见 §11.5)。
  67 cell 终裁:**vs U2r(重放,已不可实现)聚合 −0.13%(median
  +0.05%),4 击穿(最深 −8%,§11.5 全列);vs λ 回滚线 39 胜/
  18 负、聚合 +5.0%**。裁决:转正 MOSAIC-U2b,U2r 与 oracle 同归
  参考线(缓存 `u2r_replay_ref`);不回滚。**遗留偏差如实注明**:
  λ-U38 的归档数字同样未计其现实同步成本(接收端测速 + 每波 λ 分配
  下发,与通道 B 同构),x42 的 42:13 与本轮的 39:18 都**轻微偏向
  λ**;通道 B 的 L 扫描(整波陈旧仅伤 0.3-6.4%)给出该偏差的量级
  上界,不改变方向。

### 11.1 立足点修订(审计结论,取代 §3-R4 的原表述)

原 R4"共享干扰只有跑起来才可测"混淆了两种未知:**自己分配造成的
干扰**(可记账)与**外生不确定**(降速/干扰/先验过时,须测)。
λ-EWMA 学两者之和的代价:信号自回声/删失(→ §4.4 三补丁+floor)、
per-dst 分解丢跨 dst 耦合(→ E3/E4/F2/F3b/PACE/PC/jdist 补丁族)。
修订后的设计定理:**自致部分显式记账并在线朝割界水平化;测量只保留
外生轴,且以异常检测器(而非连续估计器)形态存在;账目传播只假设
有界陈旧,不假设同步**。

### 11.2 设计:MOSAIC-U2b = 账本核 + 双通道广播 + 决策错峰

```
MOSAIC-U2b = -mosaic_corridors C(wiring 定,同 §10.4)
           + -mosaic_waves W(同)
           + -mosaic_ledger                     # 账本核(rail/pod;
                                                #   cube/rail2p 回退 λ)
           + -mosaic_ledger_sync snap -mosaic_sync_us 100  # 通道 A
           + -mosaic_lg_bcast -mosaic_lg_blag_us 2         # 通道 B
           + -mosaic_lg_stagger_us 3            # 决策错峰(8 槽)
           + -mosaic_pair_color_auto            # PC 三重门不变
           + -mosaic_cold_pace 1                # PACE 三门不变
（纯 ECMP fabric 兜底档:+ -mosaic_lg_spineagg,价目见 §11.5)
```

**账本核**(x41,`scheduler/mosaic_schedule.h`):每 GPU NIC↑/NIC↓/
NV↑/NV↓ + 每 (leaf,spine) 上/下段各记 outstanding;波配额 24 步贪心
注水到"落入后最大投影完成最小"的走廊(级带 5% 内取短链,发射按
1MB 切片);prior 与 runtime 统一(t=0 空账本同一注水);λ/censoring/
trust/anti-windup/floor 全套退役。**NIC 域异常检测器**:属主本地忙时
聚合吞吐,<0.45×额定连续 3 折才锁定,quorum≤2 门(相关性塌陷=拥塞
态冻结),恢复即回额定。

**分布式协议(x44 定稿;取代 x42 的"确定性重放"表述)**:
- **通道 A(实测)**:资源属主本地计数(discharges + 检测器速率),
  每 **100µs** 发布(每 GPU 32B;细分 fabric 账时含 per-spine 归属
  ≈64B)。单写者、单调、非阻塞——账本里"精确可记"的部分。
- **通道 B(策略)**:每个目的地在 75% 触发点(发射下一波时)广播
  **本波的走廊分配**(实测 n32/64/256 = 0.3/1.0/2.6KB,亚线性);
  接收方只存各目的地**最新一份**;charges 视图 = 自己的决策(活)+
  他人最新发布。单写者 ⇒ **结构上不可能分叉**;陈旧 ≤ 一波 + 网络
  延迟。旧的"确定性重放"被 fork 探针证伪(§11.0),已废弃。
- **决策错峰**:波前的多个目的地在纳秒级内同时触发,任何非零广播
  延迟都会让它们互相看不见(实测 nv90 +6-12%,对延迟量 1-5µs 不
  敏感 = 阶跃);解法 = slotting:按 (dst mod 8)×3µs 等自己的槽位、
  读最新发布再决策,波延迟代价 ≤21µs(主线波长 570µs 的 3.7%)。
- **fabric 账目的可观测性前提(显式声明)**:per-(leaf,spine) 细分
  在**发端可选 spine 的 fabric**(WCMP/源路由,AI fabric 常见)下
  完全可实现——charge 是声明(通道 B 原样携带),discharge 由收发
  两端按链元数据归属(通道 A);**纯 ECMP 哈希 fabric 看不见 spine
  选择**,此时用 `-mosaic_lg_spineagg` 兜底(leaf/rail 交换机自有
  聚合计数 + per-dst φ 外循环反馈),实测价目:主线 ≈0(nv90 甚至
  −4.5%),pod 高压 −14~−26%、降速 −10~−15%(§11.5)——细分账目
  是真值钱的,兜底档按 fabric 能力选,不逐 cell 调。

**为什么是这一档**(x42 谱系 + x44 扫描):阻塞式逐波 allgather
(strong)只在 ≤20µs 可行;完全不同步(none)降速检测失效;通道 A
snap 100-500µs ≈ oracle(±1%);通道 B 整波陈旧仅伤 0.3-6.4%
(策略是声明不是计数,窗宽得多),真正的坑是同波触发的群体失明,
由错峰解决。

### 11.3 主线 SOTA 表(flex_rail × nv;**双通道协议实测数字**)

| nv | bound | **MOSAIC-U2b** | vs FAST | vs λ(U38) | U2r(重放,参考) |
|---|---|---|---|---|---|
| 2400 | 3327 | **3393.3** | **1.008×**(3418.8¹) | **1.033×** | 3400.6 |
| 720 | 4532 | **4754.2** | 1.066× | **1.012×** | 4683.1 |
| 600 | 5229 | 5490.3 | 1.091× | 0.991× | 5400.9 |
| 480 | 6181 | 6436.0 | 1.145× | 0.978× | 6298.8 |
| 420 | 6798 | 7271.4 | 1.203× | 0.982× | 7195.6 |
| 360 | 7556 | 7955.8 | 1.273× | 0.998×(平) | 7818.5 |
| 180 | 11330 | **11866.6** | 1.706× | 0.987× | 11948.2 |
| 90 | 15105 | **16773.3** | 2.413× | 0.939×(税点) | 16782.5 |

¹ FAST = 33 组最强口径。**2400 的翻正在全可实现协议下保持**
(1.008×);180/90 比重放参考还略快(错峰的副作用是良性错开)。
中段(480-600)对 λ 的小负 ≤2.2% 是分布式税的落点;90 仍为已知
税点(§11.5)。U2r 列 = 已被 fork 探针证伪的重放版,仅作参考
(缓存 `u2r_replay_ref`),禁止作声明基准。

### 11.4 全域结果(双通道协议实测;对手 = 冻结归档)

- **非通告降速**:hot×4@2400 = **4131.6**(vs λ-C6 8892 =
  **2.15×**,vs FAST 3.22×);cold×4 = 3443.6(比重放参考还快
  10%);hot×2/cold×2@2400 = 4076.3/3395.8(降速族 8 点里 7 点
  ≥ 重放参考,唯 hot×4@2400 −5.0%,§11.5)。**遥测是检测的
  前提**:none 模式下 hot×4 退回 ~8600(≈λ)——"能发现别人把
  你的网卡拖慢了"本身就是 100µs 遥测买来的。
- **均匀+瘦身交换机(M3 保险费)**:tp60@2400 = 2413.8(vs λ-C4
  1.31×);uniform@360/@90 = 2431.2/4111.2(@360 反超 FAST;@90
  见 §11.5 击穿行);uniform@2400 = 2432.3(vs 重放参考 1.017×)。
- **漂移**:stale prior 与 fresh 逐位相同(2553.2)且快于归档 adapt
  ——账本不读先验,结构免疫在广播协议下原样保留。
- **多热点/尺寸**:khot4@360 0.979×/khot8@90 1.006×(vs 重放参考;
  vs λ 全胜)、m512k@90 = **4588.4**(x42 最深税点 0.80 → 1.104×
  ——错峰的错开副作用把陈旧失配吃掉一半)、m8m@360 0.993×。

**拓扑覆盖分表(v7.2 新增;此前主表只展开 rail 主线,现把基准集
里其余拓扑/形态的协议实测列明)**:

| 族 | cell | **U2b** | vs U2r(重放参考) | vs λ |
|---|---|---|---|---|
| pod(C=5) | 2400/up720 | **6766.0** | **1.018×** | 1.003× |
| pod | 360/up720 | 8181.2 | 0.975× | 0.987× |
| pod | 360/up360(4:1) | **13568.2** | **1.004×** | 1.020× |
| pod | 90/up720 | **16969.6** | **1.027×** | 0.910×(税族) |
| host 形态 | 8x4@2400/360/90 | 6716.6/8573.1/15364.2 | 1.002/0.978/0.999× | 1.093/1.000/1.036× |
| host 形态 | 2x16@2400/360/90 | 1716.2/8801.8/22837.3 | 1.007/0.992/1.009× | 1.102/0.892/1.057× |
| 规模 | n64@2400/360 | 7297.2/13238.4 | 1.002/1.000× | 1.087/1.081× |

**覆盖边界(如实)**:账本(含本协议)只支持 rail/pod 两种 wiring
(`scheduler/mosaic_schedule.h: set_ledger` 的守卫);cube-mesh 与
rail2p 上 `-mosaic_ledger` 自动关闭、整套回退 λ 核——这两种拓扑的
现行数字仍是 §9/§10 的 λ 口径,账本化是开放项(§11.5)。

- 全表:`experiments/mosaic_x1/bench_runs/u2bl.csv`(每 cell 界
  核对、vs 重放参考、vs 最强其它两列);重放版全表存档
  `bench_runs/sota_u2r.csv` + `u2b.csv`(全规格 spineagg 版)。

### 11.5 代价表(v7.2:x44 门的 4 击穿全列 + 兜底档价目)

**vs U2r(重放参考)的 4 击穿(<0.95)**:

| cell | 比率 | 归因 |
|---|---|---|
| seed44@360 | 0.920 | 种子族波动(全规格 spineagg 版此格反而 1.012×;±8% 摆动,饱和边缘的调度相位敏感) |
| persrc@90 | 0.926 | 错峰在深饱和区反而增同步性(bcast 单独 = 1.000 逐位级;x42 羊群律再现——该区"看得越齐越挤") |
| uniform@90 | 0.940 | 同上(bcast 0.895、+错峰 0.940:错峰这里是净帮助,残余是广播陈旧本身) |
| deghot4@2400 | 0.950 | 降速绕行的决策对 charge 一拍陈旧最敏感(检测本身完好:vs λ 仍 2.15×) |

对比 x42 时代的 8 击穿(最深 0.80):x44 收敛到 4 个、最深 0.92,
且 m512k@90 从 0.80 修复到 1.104×(错峰的天然错开吃掉陈旧失配)。
**vs λ 的税族不变**:main@90 0.939、pod@90 0.910、seed43@90 0.915、
zipf2@90 0.921 等 NV 饱和族(18 负全在此),λ 直接测饱和耦合速率的
残余正当领地。

**兜底档价目(纯 ECMP fabric,`-mosaic_lg_spineagg`)**:主线 ≈0
(main@90 甚至 −4.5%);**pod 高压 −14~−26%、降速 −10~−15%**
(`bench_runs/u2b.csv` 全表)——per-(leaf,spine) 细分账目在这些
regime 是真值钱的;能选 spine 的 fabric 就不该放弃它。

**一个反直觉洞见(x42,x44 两次复证)**:深饱和区"看得越齐越挤"
——none@90 曾全场最佳、x44 的错峰在 persrc@90 反而致损、blagw=2
(两波陈旧)在 90 好于半波。饱和区的正解方向是**有意的视图局部化/
抖动**而非更精的同步。

开放项:①饱和区第二信号(pause/ECN 计数按资源)或有意视图抖动;
②cube/rail2p 账本化(现回退 λ,§11.4 覆盖边界);③n≥256 全套
(通信量已测,性能门未跑);④错峰槽位的拓扑感知分配(现为
dst mod 8)。

### 11.6 分布式实现路线(为什么这些不是空头支票)

- **有界陈旧(SSP)**:参数服务器界的标准做法——允许读旧 ≤s 步的
  状态,收敛有证明;我们的 snap-T 就是它的网络遥测版,且实测
  T≤500µs 内质量平坦。
- **带内遥测(INT/HPCC)**:逐跳队列深度捎在数据包头里已是部署过的
  技术(HPCC,SIGCOMM'19);discharge 计数与它同型,捎带成本≈0。
- **确定性重放(x44 起:已废弃,如实记录)**:v7.1 曾以"SPMD 同
  输入同算法"支撑 charges 免传;x44 的 fork 探针证明注水的输入还含
  分布式状态(账本快照/剩余量/波时机),视图差一拍 → 11-39% 字节/波
  分岔且累加不自愈。替代 = 通道 B 策略广播(下条)。
- **策略广播(通道 B)= 参数服务器的"发布最新参数"**:每目的地
  每波发布一份分配向量(实测 n32/64/256 = 0.3/1.0/2.6KB,亚线性;
  n256 全量互广 ≈0.4% NIC 带宽),接收方存最新——单写者 + 幂等,
  这是 KV/参数服务器语义,不需要共识;整波陈旧实测只伤 0.3-6.4%。
- **决策错峰 = slotting/带槽退避**:以太网时代的老技术;槽位由
  rank id 静态决定(dst mod 8 × 3µs),无协调开销,波延迟 ≤21µs。
- **规模账**:通道 A 每 GPU 每 100µs 发布 32-64B(n=1024 全量互换
  ≈1.4-2.8% NIC 带宽,INT 捎带近零);通道 B 每波 0.3-2.6KB/GPU
  (随波长摊薄,n256 实测 0.4%);两通道皆可走 allgather 树/捎带。
- 对照:FAST 被接受的部署形态是中心构造 111ms@256 再分发——我们的
  "周期 64B 遥测 + 每波 KB 级策略广播"严格更轻。

### 11.7 复现

```bash
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail \
  -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 \
  -scheduler mosaic_schedule -mosaic_corridors 6 \
  -mosaic_ledger -mosaic_ledger_sync snap -mosaic_sync_us 100 \
  -mosaic_lg_bcast -mosaic_lg_blag_us 2 -mosaic_lg_stagger_us 3 \
  -mosaic_pair_color_auto -mosaic_cold_pace 1 \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
# 兜底档(纯 ECMP fabric):+ -mosaic_lg_spineagg(价目 §11.5)
# 消融:去 -mosaic_lg_* 三旗 = U2r 重放版(仅参考,fork 已证伪);
#   -mosaic_lg_forkprobe 复现分叉率;-mosaic_lg_blagw {0.5|1|2} 陈旧
#   扫描;-mosaic_ledger_sync {oracle|strong|none} 同 x42(oracle 仅
#   上界参考,禁止作声明基准);去 -mosaic_ledger = 旧 λ 方案(逐位)
# 一键全套:NAME=x SCHED=mosaic_schedule FLAGS="..." script/bench_scheme.sh
```
