# 34 组:O1 波内配对着色(pair coloring)+ O3 λ 守卫(负结果)

**报告**:`Note/explore/2026-0731-x34-pair-color.md`(机制数学、pc×R
机理、PC × 其他 wiring 与着色深度探针)。

**日期**:2026-07-31。**批日志**:`cache/nonuniform/x34_opt.log` /
`x34b_opt.log` / `x34c_opt.log`(每份头部记录二进制 md5)。所有数字
validation PASSED、queue_drops 0。

## 机制

**O1 `-mosaic_pair_color K`**(`mosaic_schedule.h`):把 epoch 编码从
`wave*4+hop` 扩为 `(wave*4+hop)*K + (d−s) mod K`。数学上这是把 demand
的二部图按 Z_n 的 Cayley 表 `L(s,d)=(d−s) mod n` 分解成 n 个循环置换:
同一优先级层内每 src 恰好偏好一个 dst(Latin square),即 FAST 全局
permutation 协调的"软"形式——min-epoch 只是偏好不是闸,fallback 仍
work-conserving,不牺牲 skew 稳健性。uniform demand 时该分解恰为等权
BvN 分解,这就是它复现 FAST 优势的原因(M1 翻盘)。

**`-mosaic_pair_color_auto` 自动门**:统计量 = 对角线和的 CoV
(`diag_cov`)。含义:着色给每条对角线等份优先级时间片,对角线和相等
⇔ demand 可分解为等权 BvN 混合 ⇔ 时间片与负载对齐。hot-dst zipf 的列
在每条对角线上恰出现一次 → 列 skew 相消(实测 diag_cov 0.011-0.033);
persrc 0.237 → 门关。阈值 0.1。

**O3 `-mosaic_guard ε`(负结果)**:λ 更新前预测再平衡收益
`1−1/max_c(λ_c/share_c)`,低于 ε 冻结。实测所有点(含 nv90 伤害点)
与 base 逐位相同或 ±0.15%——M3 的探索保险费不由 λ 漂移承载(floor
才是),守卫无肉。flag 保留默认关,不推荐。

## 结果(vs 21 组 C6 归档 + SOTA 配置)

见 `results.csv`。要点:

- **M1 翻盘**:uniform@360 2504(vs FAST 2622,0.95×→**1.047×**);
  zipf1.5@2400 4925(vs FAST 5132,0.98×→**1.04×**);uniform@2400
  0.92×→0.98×;persrc 门关回落 base(0.95× 不变)。
- **主线 8 nv 回归**:7/8 点改善或持平(360/180 −5.7/−5.8%),唯
  600 +1.9%。
- **SOTA 组合**:480/600/720 pc+PACE 与 90 pc+R+PACE 四点刷新 SOTA
  (6296/5441/4812/15528);**负交互**:pc×R 在 2400(+9.5%)与 180
  (+2.3%)——SOTA 表按点推荐,2400 保持 C6+R 不加 pc。
- 已知边界:seed44@2400 +1.6%(噪声带,门按结构不按 seed);其他
  wiring(pod/rail2p/cube)未测。
- **pc×R 机理(35 组补测,results.csv isolate_* 行)**:@2400 组件
  隔离显示冲突全部来自 **E3 skew_epoch**(pc+E3 3726 ≈ pc+完整 R
  3727;pc+ramp_cold 3495 = 纯 pc;pc+rewaterfill 3501)。E3 的波位
  降级把被降 dst 从其着色层抽走,破坏 Latin square"每层每 src 恰一
  dst"的对齐。"R 去 E3"与 pc 组合在 2400/180/90 无新 SOTA
  (3501/12427/15830)→ 规则:E3 与 PC 按 regime 二选一,PC 与
  ramp/rewaterfill/PACE 正交可叠。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail \
  -flex_nv_gbps 360 -packet_size 1024 -chunk_size 2097152 \
  -message_size 2097152 -scheduler mosaic_schedule -mosaic_corridors 6 \
  -workload A2A -all_to_all -store_forward -quiet -mosaic_pair_color_auto
# 批脚本:script/x34_opt.sh / x34b_opt.sh / x34c_opt.sh
```
