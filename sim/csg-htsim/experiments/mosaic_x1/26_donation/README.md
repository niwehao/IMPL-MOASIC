# 26_donation — 捐赠晋升(D2)与 skew 感知优先序的执行期探索

问题:hot-dst 下 MOSAIC 距割界仍有 +7.8~11.9%(nv90/180/360);原愿景的
"捐赠晋升 D2"(完成事件把优先级转给长尾)未实现(报告 v5 §8.4)。
本组用 trace 诊断定位损失,再实现 5 个默认关的机制逐一消融。

## 诊断(x26_trace_nv{90,180,360}.log,`-mosaic_trace`)

- 尾段(倒数第二个 dst 完成后 hot 独跑,占总时长 45~48%)**已贴界**
  (nv90 尾段速率 = 100.7% 组合 ingress 割界)——完成触发式捐赠瞄准的
  是没有损失的阶段。
- 损失集中在**争用期**:nv90 前 3320µs hot ingress 仅 82%(中段 89%)。
  三个机制:①within-epoch 平分不按 demand 加权;②censoring 规则误杀
  hot 的 NV 走廊(2-hop 链 hop2 未及阈值,λ_rx 0.32→floor);③2-hop
  chunk 串行化(solo oracle:hot 单独跑 15318 vs B1 15104,+1.4% 固有)。
- 可回收窗口 = base 16276.9 − solo 15318 = 959µs @nv90。

## 机制(全部默认关;`scheduler/mosaic_schedule.h`)

| flag | 机制 | 针对 |
|---|---|---|
| `-mosaic_donate_window` | E1 完成事件给最大残量 dst 提前放一波 | 尾段窗口(诊断预测无效) |
| `-mosaic_donate_rewaterfill` | E2 完成事件对未完成 dst 按残量重注水直设 λ(事件锚定) | 病灶② |
| `-mosaic_skew_epoch K` | E3 生成期按 B1 slack 给 dst 降 K 波 epoch 优先序;t_max<2×median 自禁用 | 病灶① |
| `-mosaic_hop_progress` | E5 censoring 修正:relay hop 有推进的走廊保留信念 | 病灶②(测量层) |

(E4 流体相位计划在 `27_phase_plan`。)

## 主网格(zipf1.0/seed42,C=6,W=6,µs;粗体 = 每 nv 最优)

| nv | bound | base | E1 窗口 | E2 rewf | E3 skew2 | E3+E2 | static | static+skew | E5 hp | E2+E5 | E3+E2+E5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 90 | 15104 | 16276.9 | 24111 | 16173 | 15914 | **15781** | 16450 | 16407 | 16356 | 16354 | 16271 |
| 180 | 11328 | 12672.7 | 14957 | 12656 | 12596 | 12555 | 12098 | **12042** | 12377 | 12302 | 12286 |
| 360 | 7552 | 8418.4 | 8320 | 8329 | 8350 | 8365 | 8625 | 8568 | 8173 | 8105 | **8088** |
| 2400 | 3327 | 3525.8 | 3507 | 3480 | **3389** | 3405 | 3588 | 3568 | 3668 | 3581 | 3731 |

- 每 nv 最优比 base 快 3.0/5.0/3.9/3.9%;vs bound gap 7.8→4.5%、
  11.9→6.3%、11.5→7.1%、6.0→1.9%。
- **E1 大负**(90:+48%):证实尾段不缺窗口,早放波破坏 wave=implicit
  flow control。诊断预测成立。
- 单一推荐配置 **skewrf(E3+E2)**:主线四点 −3.0/−0.9/−0.6/−3.4%,
  无一点劣化;E5 按 nv 分治(帮 180/360、伤 90/2400),保持可选。

## W 交叉验证(nv90;用户观察"加波不加速"的机理闭环)

| W | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| base | 16785 | 16091 | 16277 | 16112 | 16068 |
| skewrf | 17133* | 15809 | 15781 | 15828 | 15910 |

base 从 W6→W12 赚 209µs(W 在充当事件适应的蹩脚代理);skewrf 后
W4~W12 平台化(129µs 带宽)且全部低于 base 最优——机制吸收了 W 轴
收益,加波只剩边界损耗。(*W2+K2 退化:cold 全部排在 hot 之后,勿用 W≤K。)

## Robustness(skewrf vs base)

seed43/44@{90,360}:+0.1/−0.6/−0.2/−2.3%;zipf1.5@{90,360}:−0.9/−1.8%;
zipf2.0:−0.7/−2.8%;khot2@360:+0.3%;degrade26×4@2400:−2.2%;
persrc@360 与 uniform@{360,2400}:**逐位等于 base**(E3 自禁用门 +
机制事件不触发)。queue_drops 全 0。

## 回归(handoff §6,flags-off)

C4@360=8565.57、C6 8 nv 点、taper90 C4=8475.28、pod C5=8347.98、
degrade C4=9253.09 全部逐位复现。rail2p B/4 例外:归档 9385.03 用
README 命令在**改动前代码**上也复现不出(6679.01)——23 组归档与其
复现命令原有不一致,与本次改动无关(已记录)。

数据 `results.csv`(含 `[reg_*]`/`[wsweep_*]` 行);日志
`cache/nonuniform/x26_{trace_nv*,matrix,phase2,regression,gate}.log`;
诊断 demand `cache/nonuniform/demand_n32_solo_dst26.txt`。
报告:`Note/explore/2026-0727-x26-27-donation-phase.md`。
