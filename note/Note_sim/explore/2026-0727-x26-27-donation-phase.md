# 捐赠晋升与 skew 感知优先序:MOSAIC 执行期的诊断驱动探索

v1,2026-07-27。承接 `report_mosaic_eps.md` v5 §8.4(D2 捐赠晋升未实现)
与两个用户观察:①增加 wave 数不改善表现;②demand 全局已知(hot-dst 下
每 dst 机内:跨机恒为 7:24)但方案没有利用"谁是 hot"这一信息。
实验归档:`csg-htsim/experiments/mosaic_x1/{26_donation,27_phase_plan}/`。
**本探索全部机制默认关,flags-off 行为与 v5 逐位一致(§7 回归),
25 组历史结果不受影响。**

## 0. TL;DR

1. **诊断推翻了 D2 的原始瞄准**:hot dst 独跑的尾段(占总时长 45–48%)
   **已经贴住 ingress 割界**(nv90 尾段速率 100.7%×(βv+βn)),
   "cold 完成后 hot 用不满资源"在尾段不成立;损失集中在**争用期**
   (nv90 前 20% 时段 hot ingress 只有 82%)。完成事件触发的捐赠
   (D2 原版)天然太迟。
2. **正确的杠杆是把"捐赠"前移到 t=0**:完成序从 demand 矩阵可精确预测
   (流体模型 last-2/last-3 全中),cold dst 有 5× slack——生成期直接
   给它们降优先级(E3 skew 优先序),等价于 cold 在整个执行期"预捐赠"
   端口优先权,而不是完成后才捐。
3. **结果**:每个 nv 点最优机制比 MOSAIC v5 快 **3.0–5.0%**,
   vs 割界 gap 从 +7.8/11.9/11.5/6.0% 压到 +4.5/6.3/7.1/**1.9**%
   (nv90/180/360/2400);单一推荐配置
   `-mosaic_skew_epoch 2 -mosaic_donate_rewaterfill` 主线四点全正收益,
   uniform/persrc 逐位无害(自禁用门)。
4. **用户命题的裁决**:"生成不需要动态"**在 λ 维度成立**——makespan
   最优流体计划的 λ 就是 isolated prior(时不变),相位 λ 表(E4)被
   实验否定;但 execution 层仍需要两样东西:静态**优先序**(E3,生成期
   可算)与对 feedback 测量病灶的修复(E2 事件锚定 / E5 censoring 修正,
   nv 依赖)。"加波不加速"随之闭环:W 是事件适应的蹩脚代理,机制
   到位后 W4–W12 平台化且全部优于旧最优(§6.3)。

## 1. 出发点(两条用户观察 + §8.4 缺口)

- **D2 捐赠晋升**:原愿景里接收方完成时把优先级配额转让给长尾;v5 只有
  被动弱化版(min-epoch 下落后者 wave 序号小自动赢端口,§4.5)。
- **W 不涨**:v5 §6.3 消融 W∈{1..12} 带宽 ~10%,默认 W=6 甚至非最优;
  加波没有兑现"更细粒度适应"的预期。
- **skew 已知未用**:hot-dst 生成器下每 dst 机内:跨机 = 7:24(G=8,
  31 个 src 中 7 个同 host)恒定,λ 的最优值是 demand+topo 的确定函数,
  feedback 每次"学"到的都是可闭式算出的常数;且每个 dst 是否 hot
  开跑前就知道,per-dst 独立控制环表达不出这一全局信息。

## 2. Step 0 诊断(`-mosaic_trace`,零行为改动)

新增 trace 只打印每 dst 完成事件与未完成 dst 的残量
(`mosaic_schedule.h` trace_dst_done;`x26_trace_nv{90,180,360}.log`)。
zipf1.0/seed42,C=6,hot dst = 26:

| nv | base | B1(26) | gap | 尾段占比* | 尾段速率 vs (βv+βn) | 争用期 ingress |
|---|---|---|---|---|---|---|
| 90 | 16276.9 | 15104 | +7.8% | 44.7% | **100.7%** | 82%(前 3320µs)/ 89%(中段) |
| 180 | 12672.7 | 11328 | +11.9% | 46.9% | ~93% | 类似 |
| 360 | 8418.4 | 7552 | +11.5% | 47.5% | ~94% | 类似 |

*尾段 = 倒数第二个 dst 完成(nv90 @9000.8µs)到 hot 完成(16276.9µs)。

三个争用期损失机制(证据在 `mosaic_wave dst 26` 逐波行):

- **P1 within-epoch 平分不按 demand 加权**:wave-0 所有 dst 的 chunk
  同 epoch,共享端口按流公平轮询,hot 需要的按需加权服务在最早、最关键
  的窗口拿不到。
- **P2 censoring 误杀 hot 的 NV 走廊**:hot 的 rx(2-hop relay)在争用
  下 hop2 迟于 75% 阈值启动 → "喂了字节零交付" → λ_rx 0.32→floor
  (wave1-2 rx 交付=0),而 B1 要求 nv90 下 33% 走 NV ingress。
- **P3 2-hop chunk 串行化**:relay 链 hop k+1 等 hop k 整 chunk,固有
  流水损耗。

**完成序可预测**:isolated B1 排序三个 nv 点 last-1 全中(dst 26);
流体模型(§5)nv90 last-3 (10,5,26) 全中、nv360 Spearman 0.804。

## 3. Step 1 oracle:收益上界

- **solo oracle**:只保留 dst26 列的 demand
  (`demand_n32_solo_dst26.txt`)单独跑 = **15318.1** vs B1 15104
  (+1.4%,P3+波边界的固有执行开销)。任何优先级机制的可及上界。
- **两相 oracle 无空间**:t_cold + 残量/(βv+βn) = 16330 ≥ 实测 16277,
  再次证明尾段已最优——D2 原版(完成时转让)在本系统里没有猎物。
- 可回收窗口 @nv90 = 16276.9 − 15318 = **959µs**。

## 4. 机制(全部默认关,`scheduler/mosaic_schedule.h`)

| 机制 | flag | 一句话 | 针对 |
|---|---|---|---|
| E1 窗口捐赠 | `-mosaic_donate_window` | 完成事件给最大残量 dst 提前放一波 | 尾段(预测无效,负对照) |
| E2 事件锚定 | `-mosaic_donate_rewaterfill` | 完成事件对每个未完成 dst 按**残量**重注水直设 λ(绕过 trust region 的事件跳变) | P2(策略层) |
| E3 skew 优先序 | `-mosaic_skew_epoch K` | 生成期按 B1 slack 给每 dst 一个 0..K 的 wave 降级,epoch=(wave+offset)×4+hop;**t_max<2×median 自禁用** | P1 |
| E4 相位计划 | `-mosaic_phase_plan` | 生成期流体模拟出每相 λ 表,第 k 个完成事件切相 | λ 维度的"纯计划"检验 |
| E5 censoring 修正 | `-mosaic_hop_progress` | "喂了没到"但 relay hop 有推进 ⇒ 保留信念而非归零 | P2(测量层) |

E3 就是 D2 的**前移形态**:cold 的优先级不是完成后才转让,而是从 t=0
就按 slack 让渡给 straggler——"捐赠"由事件驱动改为计划驱动,时机对了。

## 5. E4 的否定与理论归结(27_phase_plan)

流体计划器(资源 = 每 GPU NIC-ingress lane + 每 dst NV-ingress 管,
子流 max-min + progressive filling,生成开销 177ms@n32)**序预测极准**
(§2),但**份额模型结构性错误**:每 cross pair 1 条 direct 子流对
G−1 条 relay 子流,按子流计数的公平把 nv_share 推到 0.48(最优 0.33),
NIC 管欠载 ⇒ plan+static 23743(vs base 16277)、plan+feedback 18016。

这不是实现 bug 而是模型选择错误,且有构造性归结:**makespan 最优的
流体计划 = hot 从 t=0 以 βv+βn 满速(恰为 B1),cold 用剩余容量填缝
(slack ≥5×)——该计划里每个 dst 的 λ 就是 isolated waterfilling
prior,时不变**。所以"分段 λ 表"没有可赢的空间:λ 的正确答案 v5 的
prior 已经给出;计划真正该编码的是**优先序**(谁先用共享端口),
即 E3。E4 保留为负结果 + 序预测器。

## 6. 实验(26_donation;全部 validation PASSED、queue_drops=0)

### 6.1 主网格(zipf1.0/seed42,C=6,W=6,µs;粗体=每 nv 最优)

| nv | bound | base(v5) | E1 | E2 | E3(K2) | **E3+E2** | static | static+E3 | E5 | E2+E5 | E3+E2+E5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 90 | 15104 | 16276.9 | 24111 | 16173 | 15914 | **15781** | 16450 | 16407 | 16356 | 16354 | 16271 |
| 180 | 11328 | 12672.7 | 14957 | 12656 | 12596 | 12555 | 12098 | **12042** | 12377 | 12302 | 12286 |
| 360 | 7552 | 8418.4 | 8320 | 8329 | 8350 | 8365 | 8625 | 8568 | 8173 | 8105 | **8088** |
| 2400 | 3327 | 3525.8 | 3507 | 3480 | **3389** | 3405 | 3588 | 3568 | 3668 | 3581 | 3731 |

- 每 nv 最优 vs base:**−3.0% / −5.0% / −3.9% / −3.9%**;
  vs bound:+4.5% / +6.3% / +7.1% / **+1.9%**。
  nv90 回收了可回收窗口(959µs)的 52%,剩余主要是 P3(+1.4% 固有)
  与争用期未尽损失。
- **E1 是干净的负对照**(nv90 +48%):早放波破坏 wave = implicit flow
  control(v5 §6.6 的 PASSED 功臣),证实诊断"尾段不缺窗口"。
- **E5 按 nv 分治**:帮 180/360(censoring 在这里误杀走廊),伤 90/2400
  (那里 censoring 在做正确的拥塞规避)——原 censoring 补丁的价值域
  是 nv 依赖的,测量层修正保持可选 flag。
- 纯静态(static+E3,零运行时适应)只在 180 夺冠:用户命题"完全不需要
  动态"在 λ 上成立、在整体上不成立——feedback 修运行时干扰的价值
  (v5 §6.12)与其测量病灶并存。

### 6.2 单一推荐配置

`-mosaic_corridors 6 -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill`
(记 **MOSAIC-P**):主线 −3.0/−0.9/−0.6/−3.4%,无一点劣化;相对 FAST
的领先从 v5 的 2.49× 提到 **2.56×**(nv90)。

### 6.3 W 交叉验证(nv90;闭环"加波不加速")

| W | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| base | 16785 | 16091 | 16277 | 16112 | 16068 |
| MOSAIC-P | 17133* | 15809 | **15781** | 15828 | 15910 |

base 在 W6→W12 还能赚 209µs——W 在充当"事件适应"的蹩脚代理;
MOSAIC-P 后 W4–W12 平台化(带宽 129µs)且全线优于 base 任意 W。
预测(机制吸收 W 轴收益)成立。*W≤K 退化(cold 整体排到 hot 之后),
使用约束:W>K。

### 6.4 Robustness(MOSAIC-P vs base)

| 场景 | Δ | 场景 | Δ |
|---|---|---|---|
| seed43 @90/@360 | +0.1% / −0.6% | zipf1.5 @90/@360 | −0.9% / −1.8% |
| seed44 @90/@360 | −0.2% / −2.3% | zipf2.0 @90/@360 | −0.7% / −2.8% |
| khot2 @360 | +0.3%(两热点互不让,预期内) | degrade26×4 @2400 | **−2.2%** |
| persrc @360 | 0(门自禁用,逐位=base) | uniform @360/@2400 | 0(逐位=base) |

门(t_max<2×median ⇒ 全零 offset)是设计迭代的产物:未加门时 persrc
@360 劣化 +3.9%(slack 排序在无显著热点时是噪声)。加门后无害性由
构造保证。

## 7. 回归(handoff §6,flags-off ≡ v5)

C4@360 = 8565.57、C6 主线 8 nv 点(420/480/600/720 与 21 组 csv 逐位
同)、taper90 C4 = 8475.28、pod(360,leaf720) C5 = 8347.98、
degrade26×4 C4 = 9253.09 —— 全部逐位复现。
**既有归档问题(与本次无关)**:23 组 rail2p B/4@2400 的 9385.03 用其
README 命令在**改动前代码**上复现为 6679.01(git stash 验证)——
23 组归档值与复现命令原本就不一致,待 23 组补溯源。
**[2026-07-28 已结案,32 组]**:溯源 = 23 组数字产生于 X4(24 组)
**原地替换掉**的旧 RAIL2P corridor 族(同一 v5 提交内,无旧族开关)⇒
任何已提交版本都不可复现。23 组已当前代码全网格重跑并改写(九点全
≥ FAST);`report_mosaic_eps.md` §6.16/§6.17/M4/索引已加更正框;
24 组自身数字验证逐位可复现。

## 8. 结论、对 v5 文档的影响、局限

- **§8.4 D2 条目闭环**:原版"完成事件转让优先级"被诊断+E1/E2 实验
  证伪其时机(尾段已贴界);其正确形态是**生成期 slack 优先序**(E3)
  ——捐赠在 t=0 就完成。§7 M5(坏初值恢复不完全)获得部分解:E2 事件
  锚定与 E5 测量修正都是把 λ 从污染状态拉回 prior 的机制。
- **对用户两观察的回答**:①W 不涨是因为它是事件适应的代理,机制到位后
  W 曲线平台化;②skew 信息的正确用法不是动态 λ(prior 已最优)而是
  静态优先序 + 完成序预测,后者从 demand 一步可算。
- **局限**:(a) khot 多热点间无互让机制(热-热争用是下一个猎物,可与
  M1 波内配对着色合流);(b) E5 的价值域 nv 依赖,未默认开启;
  (c) P3(2-hop chunk 串行化,+1.4%)未处理,属 D1 走廊晚绑定/更小
  relay chunk 的领域;(d) E3 依赖 demand 先验,drift 场景(v5 §6.5)
  下 offset 是否该用 stale prior 未测;(e) 180 点 static+E3 仍领先
  MOSAIC-P 4%,feedback 争用期测量在中等 nv 的病灶未根治。

## 9. 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# MOSAIC-P(推荐探索配置;flags 均默认关,不影响 v5 行为)
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail -flex_nv_gbps 90 \
  -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
# 其他旋钮: -mosaic_donate_window  -mosaic_phase_plan  -mosaic_hop_progress
#           -mosaic_trace(诊断打印)
# 批日志: cache/nonuniform/x26_{matrix,phase2,regression,gate}.log(DONE 结尾)
# solo oracle demand: cache/nonuniform/demand_n32_solo_dst26.txt
```
