# 28_wave_ramp — 动态 corridor ramp 与固定 wave size 探索

日期 2026-07-27。承接 26/27 组(`Note/explore/report_donation_phase.md`)与两条用户
思路:①corridor 数从保守起步、随执行推进对 hot dst 逐步增加;②wave 由
"固定数量"改为"固定 size(数量浮动)"。完整叙事与机制分析见
`Note/explore/report_wave_ramp.md`。**全部新机制默认关,flags-off 与 v5 逐位一致。**

## 设置

- 拓扑 flex_rail n32(4×8),NIC 180,taper 180(除 taper90 行),C=6,W=6;
  demand = zipf1.0/seed42(鲁棒性行另注);chunk=padded(33554432,
  zipf1.5/2.0=67108864,khot2=16777216,uniform=2097152)。
- 新旋钮(`scheduler/mosaic_schedule.h`):
  - F1 `-mosaic_wave_bytes S`(固定 wave size,W_d=max(wmin,ceil(col/S)),
    `-mosaic_wave_min`);
  - F2 `-mosaic_wave_align`(epoch 截止对齐,offset=W_max−W_d);
  - F3 `-mosaic_ramp_start K`(全 dst 从 K 个 corridor 起步,每波+1;
    `-mosaic_ramp_event` 改为每完成事件+1);
  - F3b `-mosaic_ramp_cold K`(仅 slack≥2× 的 cold dst 限制;双重自动门:
    skew 门 + spine 可行性门;`-mosaic_ramp_cold_force` 旁路门 2)。

## 主表(completion_us;粗体 = 每 nv 最优;base/P 为历史值)

| nv | bound | base(v5) | MOSAIC-P | coldE1 | MOSAIC-R=P+coldE1 | R vs base | R vs bound |
|---|---|---|---|---|---|---|---|
| 90 | 15104 | 16276.9 | 15781.4 | 15697.8 | **15696.8** | −3.6% | +3.9% |
| 180 | 11328 | 12672.7 | 12555.3 | **11551.7** | 11855.6 | −6.4% | +4.7% |
| 360 | 7552 | 8418.4 | 8365.0 | 8153.2 | **7970.1** | −5.3% | +5.5% |
| 420 | 6798 | 7542.2 | 7669.1 | 7538.4 | **7217.9** | −4.3% | +6.2% |
| 480 | 6181 | 6589.8 | 6724.5 | =base(门) | 6724.5 | +2.0%* | +8.8% |
| 600 | 5229 | 5610.4 | 5539.1 | =base(门) | **5539.1** | −1.3% | +5.9% |
| 720 | 4532 | 4960.2 | 4920.7 | =base(门) | **4920.7** | −0.8% | +8.6% |
| 2400 | 3327 | 3525.8 | **3405.4** | =base(门) | 3405.4 | −3.4% | +2.4% |

*480/600/720/2400 上 ramp 被 spine 门关闭,MOSAIC-R ≡ MOSAIC-P(数值逐位
同,归因实验证实);480 的 +2.0% 与 taper90 的 +3.3% 是 E3+E2 在 26 组未
测点上的既有行为,非本次机制引入。**MOSAIC-R 在全部测点 ≥ MOSAIC-P**。
coldE1 单独 = 零劣化画像(门关点逐位=base):−3.6/−8.8/−3.1/−0.05/0/0/0/0。

## 关键单点

- **cold1 @90 = 15689.5**(单 flag 超越旧全组合最优 15781);
- **coldE1 @180 = 11551.7,vs bound +2.0%**(全项目该点最好成绩,
  vs base −8.8%);
- ramp_hot(用户思路①原形态)证伪:K=1 @360 = 11118(+32%),hot 从
  t=0 就需要全 corridor 集(B1 注水从第一波就要 NV ingress);
- F1 固定 size(思路②)nv 分治:仅 @360 小赢(S16+align+wm6 = 8229,
  −2.2%),90/180/2400 全负(S 越小越差,@90 S8 = 18808);F2 对齐一致
  回收 ~2% 但不翻盘;F1×F3 耦合无增益(f12S16wm6_cold1@90 = 16776.9)。

## spine 门 ablation(force 旁路;正确性 = 480/720 关、420 开)

| nv | base | coldE1_force | Δ | 门判定(t_spine/t_hot) |
|---|---|---|---|---|
| 420 | 7542.2 | 7538.4 | −0.05% | 0.97 → 开 ✓ |
| 480 | 6589.8 | 6966.1 | **+5.7%** | 1.07 → 关 ✓ |
| 600 | 5610.4 | 5564.2 | −0.8% | 1.26 → 关(保守,放弃小赢) |
| 720 | 4960.2 | 5350.6 | **+7.9%** | 1.45 → 关 ✓ |
| 2400 | 3525.8 | 3742.6(cold1) | +6.1% | 4.17 → 关 ✓ |

khot2@360 门边界(1.02)判关,放弃实测 −2.1% 的小赢(force 5996.7)。
cold2(允许 RX 绕开 spine)@2400 force = 3489.6(−1.0%)提示 K=2 变体
可在高 nv 回收部分收益,未纳入推荐(720 上 +2.5% 仍劣化)。

## 鲁棒性(MOSAIC-R vs base;persrc/uniform 门自禁用逐位=base)

| 场景 | Δ | 场景 | Δ |
|---|---|---|---|
| seed43 @90/@360 | −1.1% / −0.6% | zipf1.5 @90/@360 | −1.3% / −0.5% |
| seed44 @90/@360 | −2.1% / −2.3% | zipf2.0 @90/@360 | −0.9% / −1.1% |
| khot2 @360 | +0.3%(ramp 门关,继承 P) | degrade26×4 @2400 | −2.2% |
| taper90 @360 | +3.3%(ramp 门关,继承 P 的既有劣化) | persrc/uniform | 0(逐位) |

## 回归(改动后 flags-off)

C6@360 = 8418.42、C4@360 = 8565.57、base 4 nv 点与 MOSAIC-P@90 全部逐位
复现(x28_phase3.log reg_* 行与本文件生成前的 foreground 验证)。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# MOSAIC-R(推荐最强配置)
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail -flex_nv_gbps 360 \
  -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -mosaic_ramp_cold 1 -mosaic_ramp_event \
  -mosaic_skew_epoch 2 -mosaic_donate_rewaterfill \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
# 零风险增量(只加 cold ramp):-mosaic_ramp_cold 1 -mosaic_ramp_event
# 批日志:cache/nonuniform/x28_{explore,phase2,phase3,phase4}.log(DONE 结尾)
```
