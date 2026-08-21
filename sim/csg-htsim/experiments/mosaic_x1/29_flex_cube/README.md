# 29_flex_cube — 机内 cube-mesh + NVLink 速率异构(M2 边界实测)

2026-07-27。报告 §7.1 的 M2 把"无全交叉机内层"的拓扑列为 MOSAIC 不适用,
但从未实测。本组在新拓扑 `flex_cube`(DGX 形态:4 host × 8 GPU,机内
NDv2 式 hybrid cube-mesh 点对点链路、400/200/100G 三档混布旋钮;机间
沿用 flex_rail 的 rail 布线 nic=180/taper=180/spine=1)上实测:**MOSAIC
用新的 WIRING_CUBE corridor 族可以正常运行**,并在异构/NV 稀缺画像上
保持对 FAST 的优势。完整叙事见 `Note/explore/report_flex_cube.md`。

## 设置

- demand:zipf1.0/seed42(chunk=pad=32MB);uniform 2MB;seed43 / zipf1.5 稳健性。
- 速率画像(`-flex_cube_speeds`,16 边按 NDV2_32_CUBE_EDGES 顺序):
  主线 mix=400x6,200x6,100x4(B_in 每 GPU 500..1600G);同构 50/100/250/400x16;
  `-flex_cube_mix_rot` 旋转档位分布(rot4/8 把 hot dst 推到弱 B_in,割从
  B2 换 B1);`-flex_cube_full_gbps v` 全交叉退化。
- 下界:`tools/x1_bounds.py --cube-speeds ...`(B1/B3 用逐 dst 入射和
  B_in(d),新增机内子集割 B5;主线 3327=B2 NIC 池割)。
- MOSAIC cube 走廊族:DIRECT / RX(同轨落地+cube 路由)/ TX(cube 换轨)
  / SP_k(落 dst 第 k 快 cube 邻居,免 spine);prior = {NIC}+{入射链路}
  的 K 管注水(同机流量钉在 pinned path 末链)。

## 主表(zipf1.0/seed42,主线 mix,bound 3327µs,全部 PASSED)

| 方案 | completion_us | vs bound | vs ECMP |
|---|---|---|---|
| static_ecmp / basic | 21829.2 | 6.56× | 1× |
| TE-CCL(OPTIMAL, padded+fill) | 14864.8 | 4.47× | 1.47× |
| TACCL(OPTIMAL, padded+fill, 异构 cube 画像) | 13243.7 | 3.98× | 1.65× |
| SyCCL | SKIPPED(层级单速 + 机内稀疏网触发未修 DfsRateAlloc 崩溃,双重不可表达) | – | – |
| FAST(guard) | 5401.6 | 1.62× | 4.04× |
| MOSAIC C6(W6) | 5357.1 | 1.61× | 4.07× |
| **MOSAIC C6 W16** | **4824.2** | **1.45×** | **4.52×** |

## 速率画像扫描(MOSAIC C6 W6 / FAST,vs bound)

| 画像 | bound | FAST | MOSAIC | MOSAIC 最优变体 |
|---|---|---|---|---|
| 50x16(mesh≪NIC) | 10732 | 21839(2.04×,劣于 ECMP 持平) | 13514(1.26×) | R 12809(1.19×) |
| 100x16 | 7031 | 10920(1.55×) | 9215(1.31×) | R 8306(1.18×) |
| 250x16 | 3456 | 4399(1.27×) | 4917(1.42×) | base(W 增益消失) |
| mix 主线 | 3327 | 5402(1.62×) | 5357(1.61×) | W16 4824(1.45×) |
| 400x16 | 3327 | 3809(1.15×) | 4491(1.35×) | lane_rot 4135(1.24×) |

规律:**NV 稀缺/异构侧 MOSAIC 胜**(50x16 上 FAST 比 ECMP 还慢,MOSAIC
1.70× 领先);**同构富裕侧 FAST 的置换结构胜**(其机内步天然一对一,
契合点对点链路)。

## rot 扫描(hot dst 落在不同 B_in;MOSAIC=C6W6)

| rot | bound(项) | FAST | MOSAIC | W16 |
|---|---|---|---|---|
| 0 | 3327(B2) | 5402 | 5357 | 4824 |
| 4 | 3776(B1) | 5800 | 5353 | 5232 |
| 8 | 4161(B1) | 5620 | 6040 | 5511 |
| 12 | 3327(B2) | 5509 | 5052 | 4717 |

## 探索结论(explore 段)

- **W16 是最可迁移的杠杆**:异构画像全赢(主线 −10%,rot4/8/12、100/50、
  seed43 −5.8%、zipf1.5 −5.5%),同构 250/400 劣化(+7%/+1%)——
  波数增益随机内异构度增长(细波化解慢链路 head-of-line 与 NIC 池偏斜);
  rail 上同点仅 −0.7%,是 cube 特有现象。W 扫描:6→5357, 9→5143,
  12→4934, 16→4824, 24→5028(拐点 W16)。
- **落点 NIC 是 relay 真瓶颈**:容量加权 prior 输给 uniform_init(5357 vs
  5268);`-mosaic_cube_nic_cap`(权重取 min(nic,link))+0.7% 修正,
  `-mosaic_cube_lane_rot`(lane 落点按 src 旋转)主线 −2.7%、400x16
  −7.9%,但 rot8 劣化 +11%——两旋钮默认关,记录为分治型。
- C4 反常(7233):单 lane 按链路容量过度加权;nic_cap 下回落到 5946。
- MOSAIC-R 的双门在 cube 正确工作:主线 spine 门自动关(cold 压 spine
  会铸新割),100x16 门开 −9.9%。
- 退化对拍:full50/300 vs flex_rail@350/2100 —— ECMP 逐位一致(21829.2,
  fabric 等价性 sanity);MOSAIC/FAST 在 full 模式偏慢(点对点专用链路
  vs 共享 NV 管:relay 单对上限 v 而非全管),FAST 差距小(置换结构天然
  一对一)——这是模型差异不是 bug,详见报告 §5.4。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# MOSAIC 推荐(cube 主线):
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_cube \
  -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -mosaic_waves 16 \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
# 画像/退化旋钮:-flex_cube_speeds 400x6,200x6,100x4 | 100x16 ...
#   -flex_cube_mix_rot K | -flex_cube_full_gbps V | -flex_cube_link_us 0.7
# 下界:python3 tools/x1_bounds.py --demand $D --cube-speeds 400x6,200x6,100x4
# TE-CCL:tools/teccl_generate_schedule.py --topo cache/nonuniform/teccl_cube_topo.json
# TACCL:cache/nonuniform/taccl_flex_cube_32mb/(profile: taccl_flex_cube_topo_{2,32}mb.json)
# 批日志:cache/nonuniform/x29_{online,explore,explore2,explore3}.log(DONE 结尾)
```
