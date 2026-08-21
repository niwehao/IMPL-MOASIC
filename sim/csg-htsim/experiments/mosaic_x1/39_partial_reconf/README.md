# 39_partial_reconf — OCS 端口级部分重配(async)语义:MOSAIC-O-PR vs 35 组 barrier 归档

**设置**:n=32,k=1,800G/500ns,4MB chunk,pay-initial(全部与 35 组同口径);
demand ∈ {uniform, zipf1.0, zipf2.0, persrc}(同 35 组文件),R ∈ {10,100,1000,10000}µs。
二进制:`htsim_optical`(md5 见 `cache/optical/x39_partial.log` 首行)。
**语义**:`-async` = 端口级部分重配——端口仅在自身电路变化时付 R(各端口
时间线独立),无全网 barrier;同一 phased plan 的重执行(拓扑/路径/字节
摆放不变,只换计费与执行语义);零 duty 电路不建(对计划方有利)。
barrier 侧数字**冻结引用** 35 组归档,不重跑。

## 主表(completion_us;PR = min(全部 async 候选, barrier MOSAIC-O),两边都是精确仿真评估)

| demand | R | barrier best(35 组) | **MOSAIC-O-PR** | PR 胜者 | vs barrier | vs (R+cut) |
|---|---|---|---|---|---|---|
| uniform | 10 | 975.6 | **960.6** | ron_async d31 | 1.016× | 1.46× |
| uniform | 100 | 2695.7 | **2674.7** | ron_async d13 | 1.008× | 3.57× |
| uniform | 1000 | 7496.7 | **7376.7** | ron_async d2 | 1.016× | 4.47× |
| uniform | 10000 | 20649.9 | **20409.9** | ron_async d1 | 1.012× | 1.92× |
| zipf1.0 | 10 | 5423.2 | **5408.2** | ron_async d31 | 1.003× | 1.06× |
| zipf1.0 | 100 | 7895.3 | **6620.1** | rot{1,7,25,31} | **1.193×** | 1.27× |
| zipf1.0 | 1000 | 13272.3 | **10220.1** | rot{1,7,25,31} | **1.299×** | 1.68× |
| zipf1.0 | 10000 | 22272.3 | 22272.3 | barrier 静态(兜底) | 1.000× | 1.48× |
| zipf2.0 | 10 | 12983.5 | **12688.5** | rot{1,5,31} | 1.023× | 1.00× |
| zipf2.0 | 100 | 15085.9 | **12958.5** | rot{1,5,31} | **1.164×** | 1.02× |
| zipf2.0 | 1000 | 15985.9 | **14777.3** | rot{1,31} | 1.082× | 1.08× |
| zipf2.0 | 10000 | 24985.9 | 24985.9 | barrier 静态(兜底) | 1.000× | 1.10× |
| persrc | 10 | 1583.7 | 1583.7 | barrier 匹配相位(兜底) | 1.000× | 1.48× |
| persrc | 100 | 4158.8 | **4131.9** | ron_async d12 | 1.007× | 3.57× |
| persrc | 1000 | 10004.1 | **8253.4** | ron_async d2 | **1.212×** | 4.01× |
| persrc | 10000 | 19004.1 | 19004.1 | barrier 静态(兜底) | 1.000× | 1.72× |

cut = max-line 割界(zipf1.0 5097.7 / zipf2.0 12658.0 / persrc 1058.1 /
uniform 650.1);"vs (R+cut)" 全部 ≥1.0(sanity ✓),zipf2.0@10 贴界 1.00×。
uniform 的 cut 松(必须多跳),对照 RoN Thm-1 下界更紧(35 组 ≤1.26×)。

## 读数

1. **价值窗口 = 中大 R × skew**:zipf1.0@1000 1.30×、@100 1.19×、
   persrc@1000 1.21×、zipf2.0@100 1.16×。小 R 无肉(≤1.02×,barrier 在
   均衡轮上本来近似免费);R=10ms 静态单拓扑仍最优(async 候选 0.88-0.98×,
   PR 由候选族含静态/barrier 兜底,不劣)。
2. **收益归因(消融,zipf1.0@1000)**:
   - 同计划去 barrier:RoN d=7 16494.8 → 13220.9(**去 barrier 本身 1.25×**);
   - async 计费下重选 d:oracle 从 d=7 移到 d=2(双向环)→ 10272.9;
   - 结构搜索增量:rot{1,7,25,31}+热避让 10220.1(仅再 +0.5%);
   - **持久 skeleton(c=1/2)与漏斗链全部为负**:最好 12543.7 / 14031.6
     (relay 重耦合 + 强制绕行吃掉省下的 R)。
   即:**收益 ≈ 全部来自"去 barrier + 按端口计费重选拓扑数",
   端口级"电路持久化"没有额外收益**。
3. **d=2 双向环是 async 语义下 skew 的解形态**:每对走短方向;热列的两条
   自然 ingress 边 = 内嵌的 c=2 gateway 骨架(无需钉死);每端口成本
   2R + 该端口 byte-hop 工作。实测 10272.9 vs 每端口工作界 7960(gap
   ≈2.3ms = 双向两条 15 跳传送带的灌注/前驱等待,对注入顺序完全不敏感,
   三种排序逐位同值)。
4. **对手同语义**:ron_async 列就是 RoN oracle 计划在新语义下的重执行
   (全 d 扫,对手拿到同样的 fabric 能力);Harvest θ 单拓扑 async ≡
   barrier(逐位验证,12378.4)。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_optical
bash script/optical_partial.sh          # 全网格(~15min),log = cache/optical/x39_partial.log
# 单点(PR 选择器):
python3 tools/mosaic_pr.py --demand-file cache/nonuniform/demand_n32_zipf1.0_seed42.txt \
  --reconf-us 1000 --static-plan cache/optical/hv_z10_k1.txt
# 单个构造:
python3 tools/port_plan_gen.py --demand-file ... --reconf-us 1000 --mode rot --gens 1,31 --output p.txt
./sim/tests/htsim_optical -plan p.txt -async
```

回归:legacy 路径(barrier)在本组全部代码改动后逐位复现 35 组归档
(mosaic_o 7 cell + ron d7 + selftest,见本组开发记录);EPS 侧零影响
(`htsim_optical` 独立编译单元,`sim/tests/Makefile:53-57`)。

## 补充(审稿第二轮,2026-08-03 晚)

**n=64 补点**(demand 同 35/36 组契约;`cache/optical/x39b_n64.log`):

| cell | barrier best | async best(=PR) | 倍数 |
|---|---|---|---|
| zipf1.0 n64 R=100 | 14245.5(mosaic_o=ron d46) | **11902**(ron_async d25) | **1.20×** |
| zipf1.0 n64 R=1000 | 43515.7(63-d oracle) | **20589.6**(rot{1,63}) | **2.11×** |
| persrc n64 R=1000 | 35737.4(mosaic_o) | **20098.7**(ron_async d7) | **1.78×** |

窗口随 n 变宽在三个 cell 成立;n64 上 rot 双向环只在 zipf 大 R 是解形态
(persrc/中 R 由 ron_async 中等 d 接管,PR 候选族含两者)。

**在线 duty 反馈 + 运行中漂移(`-pr_online`,B0/B2)**:双向环合成在
partial fabric 上原生执行(无相位、无计划文件),每条电路持有到 duty
排空,完成事件驱动;漂移(Δ=0.5×seed44,结构性)到达时**局部吸收**:
在开电路抬 quota(零额外 R),已关电路只对涉事端口重开(phase-2/3
二次 session,方向类相序防死锁)。无漂移 parity:10272.9 与计划版
**逐位相同**。R=1000:

| 注入时刻 | merged | vs 串行 async 16717.0 | vs 串行 barrier 20689.6 |
|---|---|---|---|
| 3000µs | **12922.6**(reopens 0) | **1.29×** | 1.60× |
| 5000µs | 14644.7(reopens 0) | 1.14× | 1.41× |
| 8000µs | 19695.2(reopens 64) | 0.85× | 1.05× |
| 9500µs | 16913.3(reopens 64) | 0.99× | 1.22× |

**crossover 边界**:fwd 轮尚在(≤~5ms)时吸收近乎免费;fwd 已关的晚期
漂移要付整轮重开(64 端口对 ×R),劣于当作新 collective 串行——在线
策略带"晚期退化为串行"门(吸收 vs 串行的估算比较,同一精确评估授权)。
R=100 merged@3000 = 11459.4 vs 串行 13117.0 = 1.14×。

**gap 归因探针**:rot{1,31} 在 delay=0(10554.0,−5µs)与 chunk=256KB
(10454.3,−1%)下几乎不变——2.3ms gap 是**方向传送带的前驱负载链**
(串行 store-and-forward 前驱,结构性),与传播时延/chunk 粒度/注入
顺序均无关;"灌注"旧称已弃用。
