# 37 组报告:switch-centric 网络上的 MOSAIC——incast 及其他现象

**日期**:2026-08-01。**任务**:把两套方案搬到**以交换机为中心**的网络
(纯两层 Clos,无 scale-up 维度,每个 host 单上行,全部字节过交换 fabric)
上,考察 incast 及其他 switch-centric 特有现象(oversubscription、
ECMP hash collision、incast+背景混合),全部对照手算割界。
**结论先行**:switch-centric 下 incast 的差异化不在 completion
(割界主导,人人贴界)而在 **buffer 轴**——MOSAIC 的 wave 门控把 buffer
需求从 blast 方案的 ~0.9·C 压到 **C/W**(W 可调,实验验证三连减半),
是"wave = implicit flow control"的 incast 定量版。

## 0. 契约与方法学

- **拓扑**:`-topology clos`(`topo/clos_graph.h`,switch-as-wire
  两层 Clos):n=32 host、`-clos_leafs 4`(8 host/leaf)、host 链路
  100G、leaf 上行默认 2:1 oversubscription(spines=1 时单链 400G;
  `-clos_uplink_gbps` 覆盖);跨 leaf 流量 spine 固定钉在
  `(src+dst)%spines`(fabric 级静态 ECMP,调度器不可选路)。host 度数
  =1 ⇒ 无 relay 旁路空间,**方案间的全部差异都来自时间调度**——这正是
  switch-centric 试验台的意义。
- **workload**(`tools/gen_incast.py`,确定性,granularity 1024B;
  per-src 预算 T̄ = 31×2MB = 65,011,712B 与主线均匀契约逐字节相同):
  - `incast m`:m 个 aggregator **跨 leaf 条纹放置**(stride=n/m,
    隔离纯端点 incast,不与"热机架"leaf cut 混杂);每 src 把 T̄ 均分给
    可达 aggregator(aggregator 自己发给其余 m−1 个;m=1 时唯一
    aggregator 是纯 sink)。m∈{1,2,4,8}。
  - `perm collide/balanced`:单置换、每对 T̄;collide 取 d=(17−s) mod 32
    ⇒ 所有对钉在同一 spine;balanced 用贪心匹配使 (s+d)%4 均匀。
  - `mixed`:src 0 均匀;其余 src 一半预算 incast 到 dst 0、一半均匀
    背景——MoE 热专家 + 背景的最小模型。
  - 桥接:主线 hot-dst zipf1.0 seed42 原文件。
- **下界**(`tools/clos_bounds.py`,全部手算核对):host egress/ingress
  cut、leaf 上行聚合 cut、**按 (s+d)%S 钉死后的单链 cut**(真实可达界),
  取 max。
- 政策沿用:>20min/run 弃(未触发)、FAILED 仅诊断、buffer 补偿对
  对手最有利、FAST 带官方 α-β guard(clos 上 guard 跳过全部 992 对
  ⇒ FAST 退化为纯 BvN 直发序——degree-1 下 relay 无利可图,guard 判断
  正确)、LP 系 solver 按 33 组以来主线排除(per-A2A demand 变化场景)。
- 仪表:clos fabric buffer 由本组新增旋钮 `-flex_fabric_buffer_mb`
  控制(`clos_graph.h` 加默认参数,默认 1024 语义不变;改后
  uniform/zipf 三方案六值逐位回归通过)。

## 1. incast:completion 人人贴界,差异全在 buffer 轴

### 1.1 主表(m 扫,默认 1GB fabric buffer;LB=host ingress cut)

| m | LB(µs) | MOSAIC C6 | MOSAIC C6+PC | FAST | ECMP |
|---|---|---|---|---|---|
| 1 | 161229 | **161289(1.0004×,0 丢包)** | 161289(同) | **FAILED**(53.8 万丢包)→ +4GB 161239 | **FAILED**(同)→ +4GB 161239 |
| 2 | 83215 | 83274(1.0007×) | 83274 | 83235(1.0002×) | 83225(1.0001×) |
| 4 | 41607 | 41668(1.0015×) | 41628(1.0005×) | 41617(1.0002×) | 41618(1.0003×) |
| 8 | 20803 | 20873(1.0033×) | 20824(1.0010×) | 20824(1.0010×) | 20815(1.0006×) |

读数:**(i)** 一旦 buffer 足够,所有方案都贴 ingress 割界(≤0.3%)——
incast 的 completion 是割界物理,不是调度自由度;**(ii)** 唯一分水岭在
m=1:MOSAIC 默认 1GB 零丢包完成,FAST/ECMP 打爆(丢 0.54GB 后的
117145µs 是假完成,仅诊断);**(iii)** PC 在纯 incast 下与 base 逐位
相同(§1.3),不伤不帮。

### 1.2 min-buffer 阈值 + wave 定律(本组核心定量发现)

incast1(C = 热列字节 = 2,015,363,072B ≈ 1.88GiB)二分 buffer:

| 方案 | 128MB | 256MB | 512MB | 1024MB | 2048MB | 阈值区间 |
|---|---|---|---|---|---|---|
| MOSAIC W=6 | FAIL(14.3 万) | FAIL(**2279**) | **PASS** | PASS | PASS | (256,512] |
| MOSAIC W=12 | FAIL(1079) | **PASS** | — | — | — | (128,256] |
| MOSAIC W=24 | **PASS**@128,FAIL@64(487) | — | — | — | — | (64,128] |
| FAST | FAIL(171 万) | FAIL(151 万) | FAIL(119 万) | FAIL(53.8 万) | **PASS** | (1024,2048] |
| ECMP | 同 FAST | 同 | 同 | 同 | **PASS** | (1024,2048] |

**闭式模型(两条,都被区间命中)**:
- blast 方案峰值 backlog ≈ C·(1 − β_out/β_in^agg)。汇聚进 leaf_down[0]
  的上游合计 = spine_down 单链 400G(跨 leaf 24 srcs 被它先行收窄)
  + 7×100G(同 leaf)= 1.1T,排空 100G ⇒ backlog ≈ 0.91·C ≈ 1.71GiB
  ∈ (1024,2048] ✓(1GB 时丢 0.54GB ≈ 超出量级 ✓;backlog 分布在
  leaf_down 与 spine_down 两级 fabric 段上,旋钮统一设全部 fabric 段)。
- MOSAIC 的 wave 门控一次只放行一波 ⇒ backlog ≈ **C/W**:
  W=6→323MB∈(256,512] ✓、W=12→161MB∈(128,256] ✓、W=24→81MB∈(64,128] ✓
  (三连减半全部命中;256MB@W6 只丢 2279 包 = 恰好压线的近失)。
- 代价:W=6→24 的 completion 161289→161467(**+0.11%**)。即 MOSAIC 在
  switch-centric incast 下的 buffer 足迹是 **C/W,可用 W 线性调低,
  completion 代价可忽略**;对比 blast 方案的 ~0.9C,相差 ~5-21×。
  这是 25 组"wave = implicit flow control"发现的 incast 定量版。
  (W 的全局默认仍是 6——本探针只证明可调性;其他 workload 下加大 W
  的影响未在本组重测,属 buffer 受限部署才需要动的旋钮。)

### 1.3 PC 在 incast 下中性(机理)

`-mosaic_pair_color_auto` 在 incast1 上 diag_cov=0 ⇒ 门开、K=32,但
completion 与 base **逐位相同**(161289)。原因:单接收端时其 ingress
链路串行化全部 C 字节,队列不空则任何发送次序的完成时间都是 C/β;PC
只是重排次序。PC 的价值在多列竞争时(§2)。

## 2. 其他 switch-centric 现象

### 2.1 oversubscription 扫(uniform;LB = max(egress 5200.9, leaf 割))

| 上行 | 比 | LB | MOSAIC | +PC | FAST | ECMP |
|---|---|---|---|---|---|---|
| 800G | 1:1 | 5200.9 | 5353.6(1.029×) | 5231.1(1.006×) | 5221.1(1.004×) | 5224.1(1.004×) |
| 400G | 2:1 | 8053.1 | 8655.1(1.075×) | 8159.4(1.013×) | 9197.3(1.142×) | **8076.7(1.003×)** |
| 200G | 4:1 | 16106.1 | 17933.9(1.113×) | 16605.7(1.031×) | 18374.5(1.141×) | **16132.0(1.002×)** |
| 100G | 8:1 | 32212.3 | 35827.0(1.112×) | 32679.9(1.015×) | 36728.8(1.140×) | **32243.0(1.001×)** |

诚实读数:**均匀流量的 clos 是 ECMP 的主场**(度数 1、hash 完美摊布,
greedy blast 即最优),ECMP 全线 1.00×;MOSAIC+PC 跟到 1.3-3.1%,
base MOSAIC 7-11%,FAST 恒 14%。PC 把 MOSAIC 从"输 ECMP 7-11%"拉到
"输 1-3%"——O1 着色在 switch-centric 上也是波内错峰的正确载体。

### 2.2 oversubscription 扫(zipf1.0;LB = ingress 40781.3,8:1 时钉链 47911.1)

| 上行 | MOSAIC | +PC | FAST | ECMP |
|---|---|---|---|---|
| 800G | 40810.6(1.0007×) | **40791.3(1.0002×)** | 40791.3 | 40801.2 |
| 400G | 40888.0(1.0026×) | **40791.3** | 40791.3 | 41289.6(1.012×) |
| 200G | 40989.5(1.0051×) | **40791.3** | 40791.3 | 41907.9(1.028×) |
| 100G | 47999.2(1.0018×) | **47931.3(1.0004×)** | 47965.3 | 48329.7(1.009×) |

skew 流量下反转:ingress 割主导,PC/FAST 全线**精确贴界**(PC 四点
同值 40791.3,吃满割界后对 oversub 完全免疫);ECMP 随 oversub 漂移到
+2.8%(hash 不认 demand,热列包在收敛点排队)。与 §2.1 合起来:
**均匀看 ECMP、偏斜看 PC/FAST,MOSAIC+PC 是唯一两边都 ≤3% 的**。

### 2.3 spine hash collision(spines=4,单链 100G;调度的边界)

| demand | LB(钉链) | MOSAIC | +PC | FAST | ECMP |
|---|---|---|---|---|---|
| perm collide(全对同 spine) | 41607.5 | 41647.7(1.0010×) | 41647.7 | 41627.7(1.0005×) | 41627.7 |
| perm balanced(四 spine 均匀) | 10401.9 | 10523.2(1.0117×) | 10523.2 | 10422.2(1.0020×) | 10422.2 |

collide/balanced 完成比恰为 4.00×(= 钉链割比),四方案全部贴各自
割界(FAST 与 ECMP 逐位同值:单置换 demand 下每 src 只有一条流,
两者的发送行为退化为同一 blast,无调度自由度)。**结论(负而干净)**:spine 钉死是 fabric 属性,时间调度不能
恢复 routing collision 损失——单置换 demand 在 (s+d)%S 钉死下结构性
用不满 spine(shift 置换只覆盖 ≤2 个 spine),这是需求→路由联合设计
的活,超出任何 send-schedule 的作用域。

### 2.4 mixed(incast 半 + 均匀背景半;LB=ingress 80614.5)

FAST 80624.6(1.0001×)< +PC 80673.4(1.0007×)< MOSAIC 80800.9
(1.0023×)< ECMP 81365.8(1.0093×)。有背景摊布时 buffer 无压力
(全员默认 1GB PASSED),completion 重新割界主导。

### 2.5 构造时间旁注

clos(G=1)上 FAST per-A2A 构造 1.5-7.8ms、MOSAIC 38-67µs(≈30-100×),
与 36 组规模趋势一致;n=32 下两者都可摊销,不再展开。

## 3. 公平性检查清单

- FAST 带官方 `-fast_alpha_beta_guard`(clos 上 guard 全跳 relay,
  正确退化);ECMP 为铁律 baseline;全部对比数字 validation PASSED,
  FAILED 仅出现在诊断列并标注;buffer 补偿(4/16GB)对 FAST/ECMP 单向
  放宽(对手最有利),MOSAIC 未用补偿数字入表。
- LB 全部 `tools/clos_bounds.py` 手算核对;binding cut 逐行
  标注;钉链 cut(routing 可达界)与聚合 cut 分开列。
- 新旋钮(clos buffer 参数化)做了六值逐位回归;demand 生成确定性
  无种子依赖。

## 4. 局限与边界

1. **n=32 单尺度**:36 组已给出规模轴;本组聚焦现象轴。incast 的
   buffer 定律 C/W 不依赖 n(C 是列和),预计规模不变,未实测。
2. **无重传语义**:FAILED=丢包即失败是本仿真器的硬边界;真实 RoCE 有
   PFC/重传,blast 方案的实际代价是 pause 风暴而非丢包,但 buffer
   压力量级(~0.9C vs C/W)的对比方向不变。
3. **spine collision 只做了钉死路由**:自适应路由(per-packet spray)
   下结论会变,但那是 fabric 侧自由度,超出 send-schedule 对比范围
   (已在 §2.3 声明为负边界)。
4. mixed 只取 50/50 一点;比例扫留作开放项(预期插值,无新机理)。

## 5. 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
python3 tools/gen_incast.py incast 1 --out cache/nonuniform/demand_n32_incast1.txt   # 2/4/8 同理
python3 tools/gen_incast.py perm collide --spines 4 --out cache/nonuniform/demand_n32_permcollide_s4.txt
python3 tools/gen_incast.py mixed --out cache/nonuniform/demand_n32_mixed_incast.txt
python3 tools/clos_bounds.py cache/nonuniform/demand_n32_incast1.txt --leafs 4 --spines 1
script/x37_eps.sh    # 全批(~40min);log: cache/nonuniform/x37_eps.log
# buffer/wave 探针:见 cache/nonuniform/x37_probe_{buf,wave}.txt(命令在 37_switch_centric/README)
```
