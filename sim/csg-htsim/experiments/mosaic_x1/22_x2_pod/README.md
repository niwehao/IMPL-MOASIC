# 22_x2_pod — X2 探索:pod 布线 + 多 spine(原 §8.2 计划,今提前完成)

**flex_pod** 拓扑(`topo/flex_pod_graph.h build_flex_pod_graph`):每台 host
的 8 块 NIC 接**同一台** host leaf;4 条 spine,每条 leaf-spine 链路 =
leaf 上行总额/4;跨机路由钉死 spine `(s+d)%4`(fabric 级 static ECMP,
热点 hash 撞车是结构性的)。X2 论点:**corridor 集按拓扑再生成**——
pod 下 TX(换轨)无意义,新族 = `DIRECT + RX-lane m`(落到 dst 机第 m 个
sibling GPU 再 NV 转发;**落点同时决定入口 NIC 和 spine**,一个杠杆摊两个
瓶颈,`mosaic_schedule.h` WIRING_POD);**两管 waterfilling prior 公式
零改动复用**。MOSAIC 用 C=5(DIRECT+4 lane),负载 zipf1.0/seed42。

## 主网格(completion_us;leaf 上行 1440=无超额,720=2:1,360=4:1)

| nv | uplink | bound | ECMP | FAST | **MOSAIC** | M/F | 距界 |
|---|---|---|---|---|---|---|---|
| 2400 | 1440 | 3327 | 17506 | 6820 | **3719** | 1.83× | +11.8% |
| 2400 | 720 | 6654 | 18250 | 13397 | **6788** | 1.97× | **+2.0%** |
| 2400 | 360 | 13309 | 20347 | 26533 | **13430** | 1.98× | **+0.9%** |
| 360 | 1440 | 7552 | 17506 | 10126 | **8487** | 1.19× | +12.4% |
| 360 | 720 | 7552 | 18250 | 14629 | **8348** | 1.75× | +10.5% |
| 360 | 360 | 13309 | 20347 | 27686 | **13827** | 2.00× | +3.9% |
| 90 | 1440 | 15104 | 17506 | 40479 | **15602** | 2.59× | +3.3% |
| 90 | 720 | 15104 | 18250 | 40479 | **15880** | 2.55× | +5.1% |
| 90 | 360 | 15104 | 20347 | 40479 | **16683** | 2.43× | +10.5% |

附加组((360,720) 点):corridor 数 C∈{1..6} = 17939/10218/8388/**8279**/
8348/8529(C=1≈ECMP sanity;C=4/5 平台);`-mosaic_uniform_init` 8701
(+4.2%)、`-mosaic_static` 8627(+3.3%);稳健性 seed43:
18226/17357/**9003**(1.93×),zipf1.5:31411/23197/**14336**(1.62×)。

## 结论

1. **MOSAIC 全 9 点赢 FAST(1.19–2.59×),且多点貼割界**(2400 系列
   +0.9%~+2%)。§8.2 预期 1.2–1.5×(vs ECMP hash 撞车)被超越:实测
   vs ECMP 1.12–4.71×,vs FAST 更大。
2. **X2 论点成立**:corridor 族按 pruning 流程对 pod 重推(DIRECT+RX-lane,
   落点=NIC+spine 双摊开),**prior 两管公式一字不改**——"方法论可迁移、
   产物按拓扑生成"从推测变实测(§7 局限 3 关闭一半)。
3. **FAST 在 pod 上全面恶化**:无超额时 6820(rail 上同点 3541)——
   共享 leaf 让它的 i→i 摊开失效一半;4:1 超额时 26533/27686
   **比 ECMP 还慢**(fabric 盲 W1 的又一实测);nv=90 恒 40479(nv 界)。
4. pod 的 C 平台在 4–5(3–4 条 lane 覆盖 4 spine 已够,C=6 过密 +3%);
   prior/feedback 各贡献 3–4%,corridor 结构贡献主体(C1→C4 = 2.17×)。

复现:`-topology flex_pod -flex_spines 4 -flex_taper_gbps {1440|720|360}
-flex_nv_gbps {nv} -scheduler mosaic_schedule -mosaic_corridors 5`;
下界 `tools/x1_bounds.py --leaf-uplink-total T`。
